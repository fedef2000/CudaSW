#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <chrono> 
#include <iomanip>

#define MATCH 2
#define MISMATCH -1
#define GAP -2
#define BLOCK_SIZE 256

// This is used to store the score of each cell, for shorter sequences it can be changed to short to save memory
typedef int score_t; 

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

/**
 * KERNEL with Shared Memory Tiling and Warp Shuffle Reduction
 */
__global__ void compute_diagonal_tiled(score_t* d_cur, score_t* d_prev, score_t* d_prev2, 
                                       const char* __restrict__ seq1, const char* __restrict__ seq2, 
                                       int m, int n, int k, 
                                       int min_r, int diag_len,
                                       int* d_max_score) {
    
    __shared__ char s_seq1[BLOCK_SIZE];
    __shared__ char s_seq2[BLOCK_SIZE];
    
    // 32 warps max per block (1024 threads / 32)
    __shared__ int s_warp_max[32]; 

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x; 
    int laneId = tid % 32; // ID inside the warp (0-31)
    int warpId = tid / 32; // ID of the warp

    int r = -1, c = -1;
    bool active = gid < diag_len;
    int my_score = 0; 

    if (active) {
        r = min_r + gid;       
        c = (k + 2) - r;
        
        s_seq1[tid] = __ldg(&seq1[r - 1]);
        s_seq2[tid] = __ldg(&seq2[c - 1]);
    } else {
        s_seq1[tid] = 0;
        s_seq2[tid] = 0;
    }

    __syncthreads();

    // --- 3. CALCOLO DELLO SCORE ---
    if (active) {
        int min_r_prev = (k + 1 < n) ? 1 : (k + 1) - n; 
        int min_r_prev2 = (k < n) ? 1 : k - n;

        int idx_up   = (r - 1) - min_r_prev; 
        int idx_left = r - min_r_prev;       
        int idx_diag = (r - 1) - min_r_prev2; 

        score_t val_up = 0, val_left = 0, val_diag = 0;

        if (r > 1 && c > 1) val_diag = d_prev2[idx_diag];
        if (r > 1) val_up = d_prev[idx_up];
        if (c > 1) val_left = d_prev[idx_left];

        char b1 = s_seq1[tid];
        char b2 = s_seq2[tid];

        int match_score = (b1 == b2) ? MATCH : MISMATCH;
        
        int s_diag = (int)val_diag + match_score;
        int s_up   = (int)val_up + GAP;
        int s_left = (int)val_left + GAP;

        my_score = max(0, s_diag);
        my_score = max(my_score, s_up);
        my_score = max(my_score, s_left);

        d_cur[gid] = (score_t)my_score;
    }

    //REDUCTION WITH WARP SHUFFLE
    
    // Reduction intra-warp
    for (int offset = 16; offset > 0; offset /= 2) {
        // __shfl_down_sync prende il valore dal thread (laneId + offset)
        // 0xffffffff attiva tutti i thread nel warp
        int neighbor_val = __shfl_down_sync(0xffffffff, my_score, offset);
        my_score = max(my_score, neighbor_val);
    }

    // Ora il thread 0 di ogni warp (laneId == 0) possiede il max del suo warp.
    // Lo scriviamo in shared memory per condividerlo tra i warp.
    if (laneId == 0) {
        s_warp_max[warpId] = my_score;
    }

    __syncthreads(); // Aspettiamo che tutti i warp abbiano scritto

    // Final reduction (only first thread in a warp)
    if (warpId == 0) {
        // Carichiamo i massimi dei warp precedenti
        int block_max = (tid < (blockDim.x / 32)) ? s_warp_max[laneId] : 0;

        for (int offset = 16; offset > 0; offset /= 2) {
            int neighbor_val = __shfl_down_sync(0xffffffff, block_max, offset);
            block_max = max(block_max, neighbor_val);
        }

        if (tid == 0 && block_max > 0) {
            atomicMax(d_max_score, block_max);
        }
    }
}

int main() {
    std::ifstream inFileA("a.seq");
    std::ifstream inFileB("b.seq");

    if (!inFileA || !inFileB) {
        std::cerr << "Error: Could not open input files!" << std::endl;
        return 1;
    }

    std::string seq1, seq2;

    if (std::getline(inFileA, seq1) && std::getline(inFileB, seq2)) {
        long long m = seq1.size(); // rows
        long long n = seq2.size(); // cols
        long long total_cells = m * n;

        // --- Sequence Allocation (Device) ---
        char *d_seq1, *d_seq2;
        cudaMalloc((void **)&d_seq1, m * sizeof(char));
        cudaMalloc((void **)&d_seq2, n * sizeof(char));
        cudaMemcpy(d_seq1, seq1.data(), m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seq2, seq2.data(), n, cudaMemcpyHostToDevice);

        // --- Diagonal Buffer Allocation (Device) ---
        score_t *d_current, *d_prev, *d_prev2;
        size_t diag_len = std::min(m, n);
        size_t diag_bytes = diag_len * sizeof(score_t); 

        cudaMalloc((void **)&d_current, diag_bytes);
        cudaMalloc((void **)&d_prev, diag_bytes);
        cudaMalloc((void **)&d_prev2, diag_bytes);
        
        cudaMemset(d_current, 0, diag_bytes);
        cudaMemset(d_prev, 0, diag_bytes);
        cudaMemset(d_prev2, 0, diag_bytes);

        int* d_max_score;
        cudaMalloc((void**)&d_max_score, sizeof(int));
        cudaMemset(d_max_score, 0, sizeof(int));

        const int total_diagonals = m + n - 1;

        std::cout << "Starting computation on " << m << " x " << n << " matrix..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < total_diagonals; k++) {
            int min_r = std::max(1LL, k + 2 - n);
            int max_r = std::min((long long)m, (long long)k + 1);
            int diagonal_dimension = max_r - min_r + 1;
            
            if (diagonal_dimension <= 0) continue;

            int gridSize = (diagonal_dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;

            compute_diagonal_tiled<<<gridSize, BLOCK_SIZE>>>(d_current, d_prev, d_prev2, 
                                                      d_seq1, d_seq2, 
                                                      m, n, k, 
                                                      min_r, diagonal_dimension,
                                                      d_max_score);
            
            score_t* temp = d_prev2;
            d_prev2 = d_prev;
            d_prev = d_current;
            d_current = temp;
        }
        
        cudaDeviceSynchronize();
        cudaCheckErrors("Execution failed");

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        int h_max_score = 0;
        cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);

        double seconds = elapsed.count();
        double gcups = (total_cells / 1e9) / seconds;

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Optimization:    Shared Mem Tiling + Reduction + Short Type" << std::endl;
        std::cout << "Max Score Found: " << h_max_score << std::endl;
        std::cout << "Time Elapsed:    " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
        std::cout << "Performance:     " << std::fixed << std::setprecision(4) << gcups << " GCUPS" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        cudaFree(d_seq1); cudaFree(d_seq2);
        cudaFree(d_current); cudaFree(d_prev); cudaFree(d_prev2);
        cudaFree(d_max_score);
    } else {
        std::cerr << "Error: Empty files." << std::endl;
    }

    return 0;
}
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <chrono> 
#include <iomanip>  // For output formatting

// Score definitions
#define MATCH 2
#define MISMATCH -1
#define GAP -2
#define BLOCK_SIZE 256

// Macro for CUDA error checking
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
 * DIAGONAL KERNEL
 */
__global__ void compute_diagonal(int* d_cur, int* d_prev, int* d_prev2, 
                                 const char* seq1, const char* seq2, 
                                 int m, int n, int k, 
                                 int min_r, int diag_len,
                                 int* d_max_score) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= diag_len) return;

    // 1. Coordinate reconstruction (r, c)
    int r = min_r + tid;       
    int c = (k + 2) - r;       

    // Indices for sequences (0-based)
    char b1 = seq1[r - 1]; 
    char b2 = seq2[c - 1];

    // 2. Calculation of previous buffer indices
    // Start row of the previous diagonal (k-1)
    int min_r_prev = (k + 1 < n) ? 1 : (k + 1) - n + 2; 
    // Start row of the diagonal before that (k-2)
    int min_r_prev2 = (k < n) ? 1 : k - n + 2;

    int idx_up   = (r - 1) - min_r_prev; 
    int idx_left = r - min_r_prev;       
    int idx_diag = (r - 1) - min_r_prev2; 

    int val_up = 0;
    int val_left = 0;
    int val_diag = 0;

    // Safe read
    if (r > 1 && c > 1) val_diag = d_prev2[idx_diag];
    if (r > 1) val_up = d_prev[idx_up];
    if (c > 1) val_left = d_prev[idx_left];

    // 3. Score Calculation
    int score = 0;
    int match_score = (b1 == b2) ? MATCH : MISMATCH;
    
    int s_diag = val_diag + match_score;
    int s_up   = val_up + GAP;
    int s_left = val_left + GAP;

    score = max(0, s_diag);
    score = max(score, s_up);
    score = max(score, s_left);

    // 4. Writing
    d_cur[tid] = score;

    // 5. Global Max Score Update (Atomic)
    // Note: atomicMax serializes access to this variable. 
    // If the score changes often, it can slow down performance. 
    if (score > 0) { // Small optimization: avoid atomics for zeros
        atomicMax(d_max_score, score);
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
        long long m = seq1.size(); //rows
        long long n = seq2.size(); //cols
        long long total_cells = m * n;

        // --- Sequence Allocation (Device) ---
        char *d_seq1, *d_seq2;
        cudaMalloc((void **)&d_seq1, m * sizeof(char));
        cudaMalloc((void **)&d_seq2, n * sizeof(char));
        cudaMemcpy(d_seq1, seq1.data(), m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seq2, seq2.data(), n, cudaMemcpyHostToDevice);

        // --- Diagonal Buffer Allocation (Device) ---
        int *d_current, *d_prev, *d_prev2;
        size_t diag_bytes = std::min(m, n) * sizeof(int); //length of the diagonal is the minimum between rows and columns

        cudaMalloc((void **)&d_current, diag_bytes);
        cudaMalloc((void **)&d_prev, diag_bytes);
        cudaMalloc((void **)&d_prev2, diag_bytes);
        
        cudaMemset(d_current, 0, diag_bytes);
        cudaMemset(d_prev, 0, diag_bytes);
        cudaMemset(d_prev2, 0, diag_bytes);

        // --- Max Score Variable Allocation (Device) ---
        int* d_max_score;
        cudaMalloc((void**)&d_max_score, sizeof(int));
        cudaMemset(d_max_score, 0, sizeof(int)); // Initialize to 0

        const int total_diagonals = m + n - 1; //number of diagonals

        // --- START TIMER ---
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < total_diagonals; k++) {
            // Row limits calculation
            int min_r = std::max(1LL, k + 2 - n);
            int max_r = std::min((long long)m, (long long)k + 1);
            int diagonal_dimension = max_r - min_r + 1; //the number of elements in a diagonal is the number of rows in that diagonal since in each row there's only one element
            
            if (diagonal_dimension <= 0) continue;

            int gridSize = (diagonal_dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;

            compute_diagonal<<<gridSize, BLOCK_SIZE>>>(d_current, d_prev, d_prev2, 
                                                      d_seq1, d_seq2, 
                                                      m, n, k, 
                                                      min_r, diagonal_dimension,
                                                      d_max_score);
            
            // Buffer swap
            int* temp = d_prev2;
            d_prev2 = d_prev;
            d_prev = d_current;
            d_current = temp;
        }
        
        // Wait for GPU to finish everything
        cudaDeviceSynchronize();
        cudaCheckErrors("Execution failed");

        // --- STOP TIMER ---
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // Retrieve Max Score
        int h_max_score = 0;
        cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);

        // GCUPS calculation (Giga Cell Updates Per Second)
        double seconds = elapsed.count();
        double gcups = (total_cells / 1e9) / seconds;

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Max Score Found: " << h_max_score << std::endl;
        std::cout << "Time Elapsed:    " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
        std::cout << "Performance:     " << std::fixed << std::setprecision(4) << gcups << " GCUPS" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        // Cleanup
        cudaFree(d_seq1); cudaFree(d_seq2);
        cudaFree(d_current); cudaFree(d_prev); cudaFree(d_prev2);
        cudaFree(d_max_score);
    } else {
        std::cerr << "Error: Empty files." << std::endl;
    }

    return 0;
}
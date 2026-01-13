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

// TILE_SIZE must be 32 for this specific Warp-Synchronous implementation
#define TILE_SIZE 32 

typedef short score_t;

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
 * TILED KERNEL (Warp Synchronous)
 * * Grid: 1D, representing the diagonal of TILES being processed.
 * Block: 32 threads (1 Warp).
 * * Logic:
 * 1. Each block identifies which Tile (t_r, t_c) it is responsible for.
 * 2. Loads Top halo (from global mem) and Left halo (from global mem) into Shared Mem.
 * 3. Loads the Sequence chunks for this tile into Shared Mem.
 * 4. Iterates through the 32x32 tile in a diagonal wavefront pattern using purely registers/shared mem.
 * 5. Writes the resulting Right edge and Bottom edge back to Global Mem for future tiles.
 */
__global__ void compute_tiled_sw(
    const char* __restrict__ seqA, 
    const char* __restrict__ seqB,
    score_t* d_bottom_edges,
    score_t* d_right_edges,
    int m, int n,
    int tile_diag_idx,
    int num_tile_cols,
    int* d_max_score) 
{
    // FIX 1: Use 'volatile' to disable register promotion on Maxwell
    __shared__ volatile score_t s_mat[TILE_SIZE + 1][TILE_SIZE + 1];
    
    __shared__ char s_seqA[TILE_SIZE];
    __shared__ char s_seqB[TILE_SIZE];

    int tid = threadIdx.x; 

    // --- Identify Tile ---
    int t_r_min = max(0, tile_diag_idx - num_tile_cols + 1);
    int t_r = t_r_min + blockIdx.x;
    int t_c = tile_diag_idx - t_r;

    int global_r_start = t_r * TILE_SIZE;
    int global_c_start = t_c * TILE_SIZE;

    // Boundary check
    if (global_r_start >= m || global_c_start >= n) return;

    // --- Load Sequence Chunks ---
    if (tid < TILE_SIZE) {
        int r_idx = global_r_start + tid;
        s_seqA[tid] = (r_idx < m) ? seqA[r_idx] : 0;

        int c_idx = global_c_start + tid;
        s_seqB[tid] = (c_idx < n) ? seqB[c_idx] : 0;
        
        // Init Boundaries in Shared Mem
        s_mat[tid + 1][0] = 0; 
        s_mat[0][tid + 1] = 0;
    }
    // Init corner
    if (tid == 0) s_mat[0][0] = 0;

    // FIX 2: Stronger Barrier before Halo Load
    __syncthreads();

    // --- Load Halos ---
    // Load Top Halo (from Tile Above)
    if (t_r > 0 && tid < TILE_SIZE) {
        if (global_c_start + tid < n) { // Check global bounds
            int idx = (t_r - 1) * n + (global_c_start + tid);
            s_mat[0][tid + 1] = d_bottom_edges[idx];
        }
    }

    // Load Left Halo (from Tile Left)
    if (t_c > 0 && tid < TILE_SIZE) {
        if (global_r_start + tid < m) { // Check global bounds
            int idx = (t_c - 1) * m + (global_r_start + tid);
            s_mat[tid + 1][0] = d_right_edges[idx];
        }
    }

    // Load Corner (Diagonal)
    if (tid == 0 && t_r > 0 && t_c > 0) {
        int idx = (t_r - 1) * n + (global_c_start - 1);
        s_mat[0][0] = d_bottom_edges[idx];
    }

    // FIX 3: Strict Barrier after Loading Halos
    __syncthreads();

    // --- Intra-Tile Compute ---
    score_t local_max = 0;

    for (int k = 0; k < (TILE_SIZE * 2 - 1); k++) {
        int min_r = (k < TILE_SIZE) ? 1 : (k - TILE_SIZE + 2);
        int max_r = (k < TILE_SIZE) ? (k + 1) : TILE_SIZE;
        
        int current_r = min_r + tid;
        
        if (current_r <= max_r) {
            int current_c = (k + 2) - current_r;
            
            // Boundary checks for computation
            if ((global_r_start + current_r - 1 < m) && (global_c_start + current_c - 1 < n)) {
                
                score_t val_up   = s_mat[current_r - 1][current_c];
                score_t val_left = s_mat[current_r][current_c - 1];
                score_t val_diag = s_mat[current_r - 1][current_c - 1];
                
                char b1 = s_seqA[current_r - 1];
                char b2 = s_seqB[current_c - 1];
                
                int match_score = (b1 == b2) ? MATCH : MISMATCH;
                int score = max(0, (int)val_diag + match_score);
                score = max(score, (int)val_up + GAP);
                score = max(score, (int)val_left + GAP);
                
                s_mat[current_r][current_c] = (score_t)score;
                if (score > local_max) local_max = (score_t)score;
            } else {
                // Ensure padding area is zeroed out for neighbors
                s_mat[current_r][current_c] = 0;
            }
        }
        
        // FIX 4: Strong Barrier INSIDE the loop
        __syncthreads(); 
    }

    // --- Write Edges ---
    if (tid < TILE_SIZE) {
        int r_out = global_r_start + tid; 
        int c_out = global_c_start + tid; 
        
        if (r_out < m) {
            d_right_edges[t_c * m + r_out] = s_mat[tid + 1][TILE_SIZE];
        }
        
        if (c_out < n) {
            d_bottom_edges[t_r * n + c_out] = s_mat[TILE_SIZE][tid + 1];
        }
    }
    
    // --- Update Global Max ---
    // Use shfl_down_sync with full mask. On Maxwell this maps to shfl_down.
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (tid == 0 && local_max > 0) {
        atomicMax(d_max_score, (int)local_max);
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
        int m = seq1.size(); //rows
        int n = seq2.size(); //cols
        
        char *d_seqA, *d_seqB;
        cudaMalloc(&d_seqA, m);
        cudaMalloc(&d_seqB, n);
        cudaMemcpy(d_seqA, seq1.data(), m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seqB, seq2.data(), n, cudaMemcpyHostToDevice);

        int num_tile_rows = (m + TILE_SIZE - 1) / TILE_SIZE;
        int num_tile_cols = (n + TILE_SIZE - 1) / TILE_SIZE;

        // Allocation for Edges
        // Right Edges: We need to store the right edge of EVERY tile.
        score_t *d_right_edges, *d_bottom_edges;
        
        size_t sz_right = (size_t)num_tile_cols * m * sizeof(score_t);
        size_t sz_bottom = (size_t)num_tile_rows * n * sizeof(score_t);
        
        cudaMalloc(&d_right_edges, sz_right);
        cudaMalloc(&d_bottom_edges, sz_bottom);
        cudaMemset(d_right_edges, 0, sz_right);
        cudaMemset(d_bottom_edges, 0, sz_bottom);

        int* d_max_score;
        cudaMalloc(&d_max_score, sizeof(int));
        cudaMemset(d_max_score, 0, sizeof(int));

        std::cout << "Starting TILED computation..." << std::endl;
        std::cout << "Matrix: " << m << " x " << n << std::endl;
        std::cout << "Tile size: " << TILE_SIZE << " x " << TILE_SIZE << std::endl;
        std::cout << "Tile Grid: " << num_tile_rows << " x " << num_tile_cols << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        int total_tile_diagonals = num_tile_rows + num_tile_cols - 1;

        for (int k = 0; k < total_tile_diagonals; k++) {
            int t_r_min = std::max(0, k - num_tile_cols + 1); //row where diagonal starts
            int t_r_max = std::min(num_tile_rows - 1, k); //row where diagonal ends
            int num_blocks = t_r_max - t_r_min + 1; //number of tiles in the diagonal

            if (num_blocks > 0) {
                // Launch one block per tile, exactly 32 threads per block
                compute_tiled_sw<<<num_blocks, 32>>>(
                    d_seqA, d_seqB, 
                    d_bottom_edges, d_right_edges, 
                    m, n, k, num_tile_cols, 
                    d_max_score
                );
            }
        }
        
        cudaDeviceSynchronize();
        cudaCheckErrors("Kernel execution");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        int h_max;
        cudaMemcpy(&h_max, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);

        double cells = (double)m * n;
        double gcups = (cells / 1e9) / elapsed.count();

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Optimization:    TILED (32x32) + Warp Sync" << std::endl;
        std::cout << "Max Score:       " << h_max << std::endl;
        std::cout << "Time:            " << elapsed.count() << " s" << std::endl;
        std::cout << "Performance:     " << gcups << " GCUPS" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        cudaFree(d_seqA); cudaFree(d_seqB);
        cudaFree(d_right_edges); cudaFree(d_bottom_edges);
        cudaFree(d_max_score);
    }
    return 0;
}
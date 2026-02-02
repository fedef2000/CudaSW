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

// TILE_DIM 32 matches warp size
#define TILE_DIM 32 

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

__global__ void compute_tile_kernel(const char* __restrict__ seq1, const char* __restrict__ seq2,
                                    score_t* d_horiz, score_t* d_vert, score_t* d_diag,
                                    int m, int n, int tile_step, int min_bx,
                                    int* d_max_score) {
    
    // 1. Coordinates
    int bx = min_bx + blockIdx.x; 
    int by = tile_step - bx;

    int dim_grid_x = (n + TILE_DIM - 1) / TILE_DIM;
    int dim_grid_y = (m + TILE_DIM - 1) / TILE_DIM;

    if (bx < 0 || bx >= dim_grid_x || by < 0 || by >= dim_grid_y) return;

    // 2. Shared Memory
    // [row][col]. Index 0 is halo. 1..32 is data.
    __shared__ score_t s_table[TILE_DIM + 1][TILE_DIM + 1];
    __shared__ char s_seq1[TILE_DIM];
    __shared__ char s_seq2[TILE_DIM];
    __shared__ int s_block_max;

    if (threadIdx.x == 0) s_block_max = 0;

    int tx = threadIdx.x; 
    int global_col_start = bx * TILE_DIM;
    int global_row_start = by * TILE_DIM;
    
    // 3. Load Sequences
    if (global_row_start + tx < m) s_seq1[tx] = seq1[global_row_start + tx];
    else s_seq1[tx] = 0;

    if (global_col_start + tx < n) s_seq2[tx] = seq2[global_col_start + tx];
    else s_seq2[tx] = 0;

    // 4. Load Halos (Boundary Conditions)
    
    // Top Halo (from d_horiz)
    if (global_col_start + tx < n) s_table[0][tx + 1] = d_horiz[global_col_start + tx];
    else s_table[0][tx + 1] = 0;

    // Left Halo (from d_vert)
    if (global_row_start + tx < m) s_table[tx + 1][0] = d_vert[global_row_start + tx];
    else s_table[tx + 1][0] = 0;

    // Corner Halo (The Fix)
    // We read from d_diag[bx]. This contains the Bottom-Right of the Top-Left Tile (bx-1, by-1)
    // which was saved by the Top Neighbor (bx, by-1) in the previous step.
    if (tx == 0) {
        if (global_col_start > 0 && global_row_start > 0) {
            s_table[0][0] = d_diag[bx];
        } else {
            s_table[0][0] = 0; // Boundary of matrix
        }
    }

    __syncthreads();

    // 5. Save 'Corner' for the Next Tile (bx, by+1)
    // The next tile needs the Bottom-Right of the Left Neighbor (bx-1, by).
    // The Left Neighbor is our current Left Halo.
    // Specifically, s_table[32][0] holds the bottom pixel of the left halo.
    // We must save this before we overwrite d_vert with our own data.
    if (tx == 0) {
        // We save the bottom-most value of the left halo (index TILE_DIM)
        d_diag[bx] = s_table[TILE_DIM][0];
    }
    // No syncthreads needed here as d_diag is not used again by this block.

    // 6. Compute Tile (Wavefront)
    int local_max = 0;

    for (int k = 0; k < 2 * TILE_DIM - 1; ++k) {
        int i = tx + 1;       // row in s_table
        int j = (k + 2) - i;  // col in s_table
        
        bool active = (j >= 1 && j <= TILE_DIM);

        if (active) {
            // Check global bounds
            if (global_row_start + (i - 1) >= m || global_col_start + (j - 1) >= n) {
                s_table[i][j] = 0;
            } else {
                score_t val_diag = s_table[i-1][j-1];
                score_t val_up   = s_table[i-1][j];
                score_t val_left = s_table[i][j-1];

                char b1 = s_seq1[i-1];
                char b2 = s_seq2[j-1];
                int match_score = (b1 == b2) ? MATCH : MISMATCH;

                int score = max(0, (int)val_diag + match_score);
                score = max(score, (int)val_up + GAP);
                score = max(score, (int)val_left + GAP);

                s_table[i][j] = (score_t)score;
                local_max = max(local_max, score);
            }
        }
        __syncthreads(); 
    }

    // 7. Write Boundaries to Global Memory
    if (global_col_start + tx < n) {
        d_horiz[global_col_start + tx] = s_table[TILE_DIM][tx + 1];
    }

    if (global_row_start + tx < m) {
        d_vert[global_row_start + tx] = s_table[tx + 1][TILE_DIM];
    }

    // 8. Reduction for Max Score
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (tx == 0) atomicMax(&s_block_max, local_max);
    __syncthreads();

    if (tx == 0 && s_block_max > 0) atomicMax(d_max_score, s_block_max);
}

int main() {
    std::ifstream inFileA("a.seq");
    std::ifstream inFileB("b.seq");

    if (!inFileA || !inFileB) {
        std::cerr << "Error: input files" << std::endl;
        return 1;
    }

    std::string seq1, seq2;
    std::getline(inFileA, seq1);
    std::getline(inFileB, seq2);

    int m = seq1.size();
    int n = seq2.size();
    
    char *d_seq1, *d_seq2;
    cudaMalloc((void **)&d_seq1, m);
    cudaMalloc((void **)&d_seq2, n);
    cudaMemcpy(d_seq1, seq1.data(), m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq2, seq2.data(), n, cudaMemcpyHostToDevice);

    score_t *d_horiz, *d_vert, *d_diag;
    // d_diag needs to be large enough for the number of block columns
    int grid_cols = (n + TILE_DIM - 1) / TILE_DIM;
    int grid_rows = (m + TILE_DIM - 1) / TILE_DIM;

    cudaMalloc((void **)&d_horiz, (n + 1) * sizeof(score_t));
    cudaMalloc((void **)&d_vert, (m + 1) * sizeof(score_t));
    cudaMalloc((void **)&d_diag, (grid_cols + 1) * sizeof(score_t)); // The fix buffer

    cudaMemset(d_horiz, 0, (n + 1) * sizeof(score_t));
    cudaMemset(d_vert, 0, (m + 1) * sizeof(score_t));
    cudaMemset(d_diag, 0, (grid_cols + 1) * sizeof(score_t));

    int* d_max_score;
    cudaMalloc((void**)&d_max_score, sizeof(int));
    cudaMemset(d_max_score, 0, sizeof(int));

    std::cout << "Running Tiled Smith-Waterman on " << m << " x " << n << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    int total_diags = grid_rows + grid_cols - 1;

    for (int k = 0; k < total_diags; k++) {
        // Calculate which blocks are active in this diagonal
        // bx + by = k
        // 0 <= by < grid_rows  =>  0 <= k - bx < grid_rows => bx > k - grid_rows
        int min_bx = std::max(0, k - grid_rows + 1);
        int max_bx = std::min(grid_cols - 1, k);
        int num_blocks = max_bx - min_bx + 1;

        if (num_blocks > 0) {
            compute_tile_kernel<<<num_blocks, TILE_DIM>>>(d_seq1, d_seq2, 
                                                          d_horiz, d_vert, d_diag,
                                                          m, n, k, min_bx, 
                                                          d_max_score);
        }
    }
    
    cudaDeviceSynchronize();
    cudaCheckErrors("Execution");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    int h_max;
    cudaMemcpy(&h_max, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Max Score: " << h_max << std::endl;
    std::cout << "Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "GCUPS: " << (double)m * n / 1e9 / elapsed.count() << std::endl;

    cudaFree(d_seq1); cudaFree(d_seq2);
    cudaFree(d_horiz); cudaFree(d_vert); cudaFree(d_diag);
    cudaFree(d_max_score);

    return 0;
}
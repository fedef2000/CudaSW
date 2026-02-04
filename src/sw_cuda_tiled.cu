#include "sw_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>

#define TILE_DIM 32 

typedef int score_t; 

// Internal macro for error checking
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- KERNEL (Updated with score params) ---
__global__ void compute_tile_kernel(const char* __restrict__ seq1, const char* __restrict__ seq2,
                                    score_t* d_horiz, score_t* d_vert, score_t* d_diag,
                                    int m, int n, int tile_step, int min_bx,
                                    int match, int mismatch, int gap, // <--- New Params
                                    int* d_max_score) {
    
    int bx = min_bx + blockIdx.x; 
    int by = tile_step - bx;

    int dim_grid_x = (n + TILE_DIM - 1) / TILE_DIM;
    int dim_grid_y = (m + TILE_DIM - 1) / TILE_DIM;

    if (bx < 0 || bx >= dim_grid_x || by < 0 || by >= dim_grid_y) return;

    __shared__ score_t s_table[TILE_DIM + 1][TILE_DIM + 1];
    __shared__ char s_seq1[TILE_DIM];
    __shared__ char s_seq2[TILE_DIM];
    __shared__ int s_block_max;

    if (threadIdx.x == 0) s_block_max = 0;

    int tx = threadIdx.x; 
    int global_col_start = bx * TILE_DIM;
    int global_row_start = by * TILE_DIM;
    
    // Load Sequences
    if (global_row_start + tx < m) s_seq1[tx] = seq1[global_row_start + tx];
    else s_seq1[tx] = 0;

    if (global_col_start + tx < n) s_seq2[tx] = seq2[global_col_start + tx];
    else s_seq2[tx] = 0;

    // Load Halos
    if (global_col_start + tx < n) s_table[0][tx + 1] = d_horiz[global_col_start + tx];
    else s_table[0][tx + 1] = 0;

    if (global_row_start + tx < m) s_table[tx + 1][0] = d_vert[global_row_start + tx];
    else s_table[tx + 1][0] = 0;

    // Load Corner
    if (tx == 0) {
        if (global_col_start > 0 && global_row_start > 0) s_table[0][0] = d_diag[bx];
        else s_table[0][0] = 0;
    }

    __syncthreads();

    // Save Corner for next tile
    if (tx == 0) d_diag[bx] = s_table[TILE_DIM][0];

    // Compute Wavefront inside Tile
    int local_max = 0;
    for (int k = 0; k < 2 * TILE_DIM - 1; ++k) {
        int i = tx + 1;       
        int j = (k + 2) - i;  
        
        if (j >= 1 && j <= TILE_DIM) {
            // Bounds check inside shared memory logic
            if (global_row_start + (i - 1) < m && global_col_start + (j - 1) < n) {
                score_t val_diag = s_table[i-1][j-1];
                score_t val_up   = s_table[i-1][j];
                score_t val_left = s_table[i][j-1];

                char b1 = s_seq1[i-1];
                char b2 = s_seq2[j-1];
                
                // Use the passed parameters instead of macros
                int score_match = (b1 == b2) ? match : mismatch;

                int score = max(0, (int)val_diag + score_match);
                score = max(score, (int)val_up + gap);
                score = max(score, (int)val_left + gap);

                s_table[i][j] = (score_t)score;
                local_max = max(local_max, score);
            } else {
                s_table[i][j] = 0;
            }
        }
        __syncthreads(); 
    }

    // Write Boundaries
    if (global_col_start + tx < n) d_horiz[global_col_start + tx] = s_table[TILE_DIM][tx + 1];
    if (global_row_start + tx < m) d_vert[global_row_start + tx] = s_table[tx + 1][TILE_DIM];

    // Warp Reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (tx == 0) atomicMax(&s_block_max, local_max);
    __syncthreads();
    if (tx == 0 && s_block_max > 0) atomicMax(d_max_score, s_block_max);
}


// --- LIBRARY HOST FUNCTION ---
int sw_cuda_tiled(const char* seq1, int len1, 
                const char* seq2, int len2, 
                SWConfig config) {
    
    // 1. Device Allocation
    char *d_seq1, *d_seq2;
    cudaCheck(cudaMalloc((void **)&d_seq1, len1 * sizeof(char)));
    cudaCheck(cudaMalloc((void **)&d_seq2, len2 * sizeof(char)));
    
    cudaCheck(cudaMemcpy(d_seq1, seq1, len1, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_seq2, seq2, len2, cudaMemcpyHostToDevice));

    score_t *d_horiz, *d_vert, *d_diag;
    int grid_cols = (len2 + TILE_DIM - 1) / TILE_DIM;
    int grid_rows = (len1 + TILE_DIM - 1) / TILE_DIM;

    size_t sz_horiz = (len2 + 1) * sizeof(score_t);
    size_t sz_vert  = (len1 + 1) * sizeof(score_t);
    size_t sz_diag  = (grid_cols + 1) * sizeof(score_t);

    cudaCheck(cudaMalloc((void **)&d_horiz, sz_horiz));
    cudaCheck(cudaMalloc((void **)&d_vert, sz_vert));
    cudaCheck(cudaMalloc((void **)&d_diag, sz_diag));

    cudaCheck(cudaMemset(d_horiz, 0, sz_horiz));
    cudaCheck(cudaMemset(d_vert, 0, sz_vert));
    cudaCheck(cudaMemset(d_diag, 0, sz_diag));

    int* d_max_score;
    cudaCheck(cudaMalloc((void**)&d_max_score, sizeof(int)));
    cudaCheck(cudaMemset(d_max_score, 0, sizeof(int)));

    // 2. Wavefront Loop
    int total_diags = grid_rows + grid_cols - 1;

    for (int k = 0; k < total_diags; k++) {
        int min_bx = std::max(0, k - grid_rows + 1);
        int max_bx = std::min(grid_cols - 1, k);
        int num_blocks = max_bx - min_bx + 1;

        if (num_blocks > 0) {
            compute_tile_kernel<<<num_blocks, TILE_DIM>>>(
                d_seq1, d_seq2, d_horiz, d_vert, d_diag,
                len1, len2, k, min_bx,
                config.match_score, config.mismatch_score, config.gap_score, // Pass params
                d_max_score
            );
        }
    }
    
    cudaCheck(cudaDeviceSynchronize());
    
    // 3. Retrieve Result
    int h_max_score = 0;
    cudaCheck(cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost));

    // 4. Cleanup
    cudaFree(d_seq1); cudaFree(d_seq2);
    cudaFree(d_horiz); cudaFree(d_vert); cudaFree(d_diag);
    cudaFree(d_max_score);

    return h_max_score;
}
#include "sw_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <utility> 

#define BLOCK_SIZE 256

/**
 * KERNEL: Diagonal Wavefront (Basic Version)
 */
__global__ void compute_diagonal_kernel(score_t* d_cur, score_t* d_prev, score_t* d_prev2, 
                                       const char* __restrict__ seq1, const char* __restrict__ seq2, 
                                       int m, int n, int k, 
                                       int min_r, int diag_len,
                                       int match, int mismatch, int gap,
                                       int* d_max_score) {
    
    __shared__ char s_seq1[BLOCK_SIZE];
    __shared__ char s_seq2[BLOCK_SIZE];
    __shared__ int s_warp_max[32]; 

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x; 
    int laneId = tid % 32; 
    int warpId = tid / 32; 

    int r = -1, c = -1;
    bool active = gid < diag_len;
    int my_score = 0; 

    if (active) {
        r = min_r + gid;       
        c = (k + 2) - r;
        
        // Load sequence characters
        s_seq1[tid] = __ldg(&seq1[r - 1]);
        s_seq2[tid] = __ldg(&seq2[c - 1]);
    } else {
        s_seq1[tid] = 0;
        s_seq2[tid] = 0;
    }

    __syncthreads();

    // --- Compute Score ---
    if (active) {
        // Mapping linear diagonal index back to 2D coordinates for previous diagonals
        int min_r_prev = (k + 1 <= n) ? 1 : (k + 1) - n; 
        int min_r_prev2 = (k <= n) ? 1 : k - n;

        int idx_up   = (r - 1) - min_r_prev; 
        int idx_left = r - min_r_prev;       
        int idx_diag = (r - 1) - min_r_prev2; 

        score_t val_up = 0, val_left = 0, val_diag = 0;

        // Fetch previous scores from global memory buffers
        if (r > 1 && c > 1) val_diag = d_prev2[idx_diag];
        if (r > 1) val_up = d_prev[idx_up];
        if (c > 1) val_left = d_prev[idx_left];

        char b1 = s_seq1[tid];
        char b2 = s_seq2[tid];

        // Use dynamic scoring parameters
        int score_match = (b1 == b2) ? match : mismatch;
        
        int s_diag = (int)val_diag + score_match;
        int s_up   = (int)val_up + gap;
        int s_left = (int)val_left + gap;

        my_score = max(0, s_diag);
        my_score = max(my_score, s_up);
        my_score = max(my_score, s_left);

        d_cur[gid] = (score_t)my_score;
    }

    // --- Reduction (Intra-Warp) ---
    for (int offset = 16; offset > 0; offset /= 2) {
        int neighbor_val = __shfl_down_sync(0xffffffff, my_score, offset);
        my_score = max(my_score, neighbor_val);
    }

    if (laneId == 0) {
        s_warp_max[warpId] = my_score;
    }

    __syncthreads(); 

    // --- Reduction (Inter-Warp) ---
    if (warpId == 0) {
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

// --- LIBRARY HOST FUNCTION ---
std::pair<int, size_t> sw_cuda_diagonal(const std::string& seq1, const std::string& seq2,
                                        SWConfig config) {

    int len1 = seq1.length();
    int len2 = seq2.length();
    
    // Track total allocated bytes
    size_t total_gpu_bytes = 0;

    // 1. Allocation
    char *d_seq1, *d_seq2;
    size_t seq1_size = len1 * sizeof(char);
    size_t seq2_size = len2 * sizeof(char);

    cudaCheck(cudaMalloc((void **)&d_seq1, seq1_size));
    cudaCheck(cudaMalloc((void **)&d_seq2, seq2_size));
    total_gpu_bytes += (seq1_size + seq2_size);

    cudaCheck(cudaMemcpy(d_seq1, seq1.c_str(), len1, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_seq2, seq2.c_str(), len2, cudaMemcpyHostToDevice));

    // Diagonal buffers: 3 full diagonal arrays
    score_t *d_current, *d_prev, *d_prev2;
    size_t diag_len = std::min(len1, len2);
    size_t diag_bytes = diag_len * sizeof(score_t); 

    cudaCheck(cudaMalloc((void **)&d_current, diag_bytes));
    cudaCheck(cudaMalloc((void **)&d_prev, diag_bytes));
    cudaCheck(cudaMalloc((void **)&d_prev2, diag_bytes));
    total_gpu_bytes += (diag_bytes * 3);
    
    cudaCheck(cudaMemset(d_current, 0, diag_bytes));
    cudaCheck(cudaMemset(d_prev, 0, diag_bytes));
    cudaCheck(cudaMemset(d_prev2, 0, diag_bytes));

    int* d_max_score;
    cudaCheck(cudaMalloc((void**)&d_max_score, sizeof(int)));
    total_gpu_bytes += sizeof(int);
    cudaCheck(cudaMemset(d_max_score, 0, sizeof(int)));

    const int total_diagonals = len1 + len2 - 1;

    // 2. Wavefront Loop
    for (int k = 0; k < total_diagonals; k++) {
        long long m_long = len1;
        long long n_long = len2;
        
        int min_r = std::max(1LL, (long long)k + 2 - n_long);
        int max_r = std::min(m_long, (long long)k + 1);
        int diagonal_dimension = max_r - min_r + 1;
        
        if (diagonal_dimension <= 0) continue;

        int gridSize = (diagonal_dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;

        compute_diagonal_kernel<<<gridSize, BLOCK_SIZE>>>(d_current, d_prev, d_prev2, 
                                                  d_seq1, d_seq2, 
                                                  len1, len2, k, 
                                                  min_r, diagonal_dimension,
                                                  config.match_score, config.mismatch_score, config.gap_score,
                                                  d_max_score);
        
        // Pointer Swapping
        score_t* temp = d_prev2;
        d_prev2 = d_prev;
        d_prev = d_current;
        d_current = temp;
    }
    
    cudaCheck(cudaDeviceSynchronize());

    // 3. Retrieve Result
    int h_max_score = 0;
    cudaCheck(cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost));

    // 4. Cleanup
    cudaFree(d_seq1); cudaFree(d_seq2);
    cudaFree(d_current); cudaFree(d_prev); cudaFree(d_prev2);
    cudaFree(d_max_score);

    return std::make_pair(h_max_score, total_gpu_bytes);
}
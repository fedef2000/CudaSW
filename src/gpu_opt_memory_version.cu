#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <chrono>
#include <iomanip>
#include <vector>

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
 * FIXED KERNEL (Triple Buffering for Bottom Edges)
 * * Nuovi Argomenti:
 * - d_bottom_corner: Punta al buffer della diagonale K-2 (per recuperare l'angolo sicuro)
 * - d_bottom_top: Punta al buffer della diagonale K-1 (per l'halo superiore)
 */
__global__ void compute_tiled_sw_triple_buf(
    const char* __restrict__ seqA, 
    const char* __restrict__ seqB,
    const score_t* __restrict__ d_bottom_top,     // Input: Diagonale K-1 (Halo Top)
    const score_t* __restrict__ d_bottom_corner,  // Input: Diagonale K-2 (Halo Corner)
    score_t* __restrict__ d_bottom_out,           // Output: Diagonale K
    const score_t* __restrict__ d_right_in,       // Input: Diagonale K-1 (Halo Left)
    score_t* __restrict__ d_right_out,            // Output: Diagonale K
    int m, int n,
    int tile_diag_idx,
    int num_tile_cols,
    int* d_max_score) 
{
    __shared__ score_t s_mat[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ char s_seqA[TILE_SIZE];
    __shared__ char s_seqB[TILE_SIZE];

    int tid = threadIdx.x; 

    // --- 1. Identifica Coordinate Tile ---
    int t_r_min = max(0, tile_diag_idx - num_tile_cols + 1);
    int t_r = t_r_min + blockIdx.x;
    int t_c = tile_diag_idx - t_r;

    int global_r_start = t_r * TILE_SIZE;
    int global_c_start = t_c * TILE_SIZE;

    if (global_r_start >= m || global_c_start >= n) return;

    // --- 2. Carica Sequenze ---
    if (tid < TILE_SIZE) {
        int r_idx = global_r_start + tid;
        s_seqA[tid] = (r_idx < m) ? seqA[r_idx] : 0;

        int c_idx = global_c_start + tid;
        s_seqB[tid] = (c_idx < n) ? seqB[c_idx] : 0;
    }

    // --- 3. Carica Halos ---
    if (tid <= TILE_SIZE) {
        s_mat[tid][0] = 0; 
        s_mat[0][tid] = 0; 
    }

    // A. Top Halo (usa buffer K-1)
    if (t_r > 0 && tid < TILE_SIZE) {
        int col_idx = global_c_start + tid;
        if (col_idx < n)
            s_mat[0][tid + 1] = d_bottom_top[col_idx];
    }

    // B. Left Halo (usa buffer K-1)
    if (t_c > 0 && tid < TILE_SIZE) {
        int row_idx = global_r_start + tid;
        if (row_idx < m)
            s_mat[tid + 1][0] = d_right_in[row_idx];
    }

    // C. Corner (usa buffer K-2 !!)
    // Questo è il fix cruciale: leggiamo dal buffer che non è stato sovrascritto
    // dal vicino sinistro nella fase K-1.
    if (tid == 0) { 
        if (t_r > 0 && t_c > 0) {
            s_mat[0][0] = d_bottom_corner[global_c_start - 1];
        } else {
             s_mat[0][0] = 0;
        }
    }

    __syncwarp(); 

    // --- 4. Calcolo Wavefront Intra-Tile (Standard) ---
    score_t local_max = 0;

    for (int k = 0; k < (TILE_SIZE * 2 - 1); k++) {
        int min_r = (k < TILE_SIZE) ? 1 : (k - TILE_SIZE + 2);
        int max_r = (k < TILE_SIZE) ? (k + 1) : TILE_SIZE;
        int current_r = min_r + tid;
        
        if (current_r <= max_r) {
            int current_c = (k + 2) - current_r;
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
                s_mat[current_r][current_c] = 0;
            }
        }
        __syncwarp(); 
    }

    // --- 5. Scrittura Output ---
    if (tid < TILE_SIZE) {
        int r_out = global_r_start + tid; 
        int c_out = global_c_start + tid; 
        
        if (r_out < m) d_right_out[r_out] = s_mat[tid + 1][TILE_SIZE];
        if (c_out < n) d_bottom_out[c_out] = s_mat[TILE_SIZE][tid + 1];
    }
    
    // --- 6. Max Reduction ---
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (tid == 0 && local_max > 0) atomicMax(d_max_score, (int)local_max);
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
        int m = seq1.size(); 
        int n = seq2.size(); 
        
        char *d_seqA, *d_seqB;
        cudaMalloc(&d_seqA, m);
        cudaMalloc(&d_seqB, n);
        cudaMemcpy(d_seqA, seq1.data(), m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seqB, seq2.data(), n, cudaMemcpyHostToDevice);

        int num_tile_rows = (m + TILE_SIZE - 1) / TILE_SIZE;
        int num_tile_cols = (n + TILE_SIZE - 1) / TILE_SIZE;

        // --- ALLOCAZIONE TRIPLE BUFFERING ---
        // Right Edges: 2 buffer (Ping-Pong è sufficiente)
        // Bottom Edges: 3 buffer (K, K-1, K-2 per preservare il corner)
        
        score_t *d_right_edges, *d_bottom_edges;
        
        size_t sz_right = 2 * m * sizeof(score_t);
        size_t sz_bottom = 3 * n * sizeof(score_t); // <--- Changed to 3
        
        cudaMalloc(&d_right_edges, sz_right);
        cudaMalloc(&d_bottom_edges, sz_bottom);
        cudaMemset(d_right_edges, 0, sz_right);
        cudaMemset(d_bottom_edges, 0, sz_bottom);

        // Puntatori di comodo
        score_t *d_r_bufs[2];
        d_r_bufs[0] = d_right_edges;
        d_r_bufs[1] = d_right_edges + m;

        score_t *d_b_bufs[3];
        d_b_bufs[0] = d_bottom_edges;
        d_b_bufs[1] = d_bottom_edges + n;
        d_b_bufs[2] = d_bottom_edges + 2 * n;

        int* d_max_score;
        cudaMalloc(&d_max_score, sizeof(int));
        cudaMemset(d_max_score, 0, sizeof(int));

        std::cout << "Starting TRIPLE BUFFERED computation..." << std::endl;
        std::cout << "Memory Usage (Edges): " << (sz_right + sz_bottom) / 1024.0 / 1024.0 << " MB" << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        int total_tile_diagonals = num_tile_rows + num_tile_cols - 1;

        for (int k = 0; k < total_tile_diagonals; k++) {
            int t_r_min = std::max(0, k - num_tile_cols + 1);
            int t_r_max = std::min(num_tile_rows - 1, k);
            int num_blocks = t_r_max - t_r_min + 1;

            if (num_blocks > 0) {
                // LOGICA BUFFER
                // Bottom: Rotazione ciclica modulo 3
                // Out: k%3
                // Top Input: (k-1)%3
                // Corner Input: (k-2)%3
                
                int idx_b_out = k % 3;
                int idx_b_top = (k + 2) % 3; // Equivalente a (k-1)%3 gestendo i negativi
                int idx_b_corner = (k + 1) % 3; // Equivalente a (k-2)%3 gestendo i negativi

                // Right: Ping Pong modulo 2
                int idx_r_out = k % 2;
                int idx_r_in = (k + 1) % 2; // (k-1)%2

                compute_tiled_sw_triple_buf<<<num_blocks, 32>>>(
                    d_seqA, d_seqB, 
                    d_b_bufs[idx_b_top],    // Input Top (K-1)
                    d_b_bufs[idx_b_corner], // Input Corner (K-2)
                    d_b_bufs[idx_b_out],    // Output (K)
                    d_r_bufs[idx_r_in],     // Input Left (K-1)
                    d_r_bufs[idx_r_out],    // Output (K)
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

        std::cout << "Max Score:       " << h_max << std::endl;
        std::cout << "Time:            " << elapsed.count() << " s" << std::endl;
        std::cout << "Performance:     " << gcups << " GCUPS" << std::endl;

        cudaFree(d_seqA); cudaFree(d_seqB);
        cudaFree(d_right_edges); cudaFree(d_bottom_edges);
        cudaFree(d_max_score);
    }
    return 0;
}
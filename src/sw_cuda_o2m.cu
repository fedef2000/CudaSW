#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "sw_cuda.h"

std::vector<int> sw_cuda_o2m(const std::string& query, 
                             const std::vector<std::string>& targets, 
                             SWConfig config) {
    std::vector<int> results;
    size_t num_targets = targets.size();
    if (num_targets == 0) return results;

    results.resize(num_targets);

    // 1. Prepare the Query (Seq1)
    // We allocate and copy this ONLY ONCE. It stays read-only on the GPU.
    int query_len = query.length();
    char* d_query;
    cudaCheck(cudaMalloc((void**)&d_query, query_len * sizeof(char)));
    cudaCheck(cudaMemcpy(d_query, query.c_str(), query_len, cudaMemcpyHostToDevice));

    // 2. Analyze Targets to determine Max Buffer Sizes
    size_t max_target_len = 0;
    for (const auto& t : targets) max_target_len = std::max(max_target_len, t.length());
    
    // Safety padding
    max_target_len += 64;

    // 3. Resource Pool Setup (Streams)
    const int N_STREAMS = 4;
    cudaStream_t streams[N_STREAMS];

    // Per-Stream Resources
    char* d_target[N_STREAMS];
    
    // Working buffers
    // d_vert depends on Query Length (Constant)
    // d_horiz depends on Target Length (Variable -> use Max)
    score_t *d_horiz[N_STREAMS], *d_vert[N_STREAMS], *d_diag[N_STREAMS];
    int* d_max_score[N_STREAMS];
    int* h_max_pinned[N_STREAMS]; // Pinned host memory

    // Calculate sizes
    size_t sz_target = max_target_len * sizeof(char);
    size_t sz_horiz  = (max_target_len + 1) * sizeof(score_t);
    size_t sz_vert   = (query_len + 1) * sizeof(score_t); // Fixed based on query
    
    // Max grid columns depends on max target length
    int max_grid_cols = (max_target_len + TILE_DIM - 1) / TILE_DIM;
    size_t sz_diag   = (max_grid_cols + 1) * sizeof(score_t);

    for (int i = 0; i < N_STREAMS; i++) {
        cudaCheck(cudaStreamCreate(&streams[i]));
        
        cudaCheck(cudaMalloc((void**)&d_target[i], sz_target));
        cudaCheck(cudaMalloc((void**)&d_horiz[i], sz_horiz));
        cudaCheck(cudaMalloc((void**)&d_vert[i], sz_vert));
        cudaCheck(cudaMalloc((void**)&d_diag[i], sz_diag));
        cudaCheck(cudaMalloc((void**)&d_max_score[i], sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&h_max_pinned[i], sizeof(int)));
    }

    // 4. Execution Loop
    for (size_t i = 0; i < num_targets; ++i) {
        int s = i % N_STREAMS; 

        // A. SYNC PREVIOUS WORK ON THIS STREAM
        cudaCheck(cudaStreamSynchronize(streams[s]));

        // B. COLLECT PREVIOUS RESULT
        if (i >= N_STREAMS) {
            results[i - N_STREAMS] = *h_max_pinned[s];
        }

        // C. PREPARE NEW JOB
        int target_len = targets[i].length();

        // Copy ONLY the target sequence (Async)
        cudaCheck(cudaMemcpyAsync(d_target[s], targets[i].c_str(), target_len, cudaMemcpyHostToDevice, streams[s]));

        // Memset working buffers (Async)
        // Reset d_vert (rows = query length)
        cudaCheck(cudaMemsetAsync(d_vert[s], 0, (query_len + 1) * sizeof(score_t), streams[s]));
        // Reset d_horiz (cols = target length)
        cudaCheck(cudaMemsetAsync(d_horiz[s], 0, (target_len + 1) * sizeof(score_t), streams[s]));
        
        int grid_cols = (target_len + TILE_DIM - 1) / TILE_DIM;
        cudaCheck(cudaMemsetAsync(d_diag[s], 0, (grid_cols + 1) * sizeof(score_t), streams[s]));
        cudaCheck(cudaMemsetAsync(d_max_score[s], 0, sizeof(int), streams[s]));

        // D. LAUNCH KERNELS
        int grid_rows = (query_len + TILE_DIM - 1) / TILE_DIM;
        int total_diags = grid_rows + grid_cols - 1;

        for (int k = 0; k < total_diags; k++) {
            int min_bx = std::max(0, k - grid_rows + 1);
            int max_bx = std::min(grid_cols - 1, k);
            int num_blocks = max_bx - min_bx + 1;

            if (num_blocks > 0) {
                // Notice: We pass 'd_query' as the first sequence for ALL streams
                compute_tile_kernel<<<num_blocks, TILE_DIM, 0, streams[s]>>>(
                    d_query,      
                    d_target[s], 
                    d_horiz[s], d_vert[s], d_diag[s],
                    query_len, target_len, k, min_bx,
                    config.match_score, config.mismatch_score, config.gap_score,
                    d_max_score[s]
                );
            }
        }

        // E. COPY RESULT BACK
        cudaCheck(cudaMemcpyAsync(h_max_pinned[s], d_max_score[s], sizeof(int), cudaMemcpyDeviceToHost, streams[s]));
    }

    // 5. Cleanup Final Jobs
    cudaCheck(cudaDeviceSynchronize());

    // Collect remaining results
    size_t start_cleanup = (num_targets >= N_STREAMS) ? (num_targets - N_STREAMS) : 0;
    for (size_t j = start_cleanup; j < num_targets; j++) {
        int s = j % N_STREAMS;
        results[j] = *h_max_pinned[s];
    }

    // 6. Free Resources
    cudaFree(d_query); // Free the shared query
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_target[i]);
        cudaFree(d_horiz[i]);
        cudaFree(d_vert[i]);
        cudaFree(d_diag[i]);
        cudaFree(d_max_score[i]);
        cudaFreeHost(h_max_pinned[i]);
    }
    return results;
}
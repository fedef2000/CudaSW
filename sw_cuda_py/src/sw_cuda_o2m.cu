#include "sw_cuda.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring> // for memcpy

// Helper for cleanup to keep code clean
void cleanup_resources(cudaStream_t* streams, char** d_target, char** h_target_pinned, 
                       score_t** d_horiz, score_t** d_vert, score_t** d_diag, 
                       int** d_max, int** h_max_pinned, int n_streams) {
    for (int i = 0; i < n_streams; i++) {
        if (streams[i]) cudaStreamDestroy(streams[i]);
        if (d_target[i]) cudaFree(d_target[i]);
        if (h_target_pinned[i]) cudaFreeHost(h_target_pinned[i]); // Free Pinned Memory
        if (d_horiz[i]) cudaFree(d_horiz[i]);
        if (d_vert[i]) cudaFree(d_vert[i]);
        if (d_diag[i]) cudaFree(d_diag[i]);
        if (d_max[i]) cudaFree(d_max[i]);
        if (h_max_pinned[i]) cudaFreeHost(h_max_pinned[i]);
    }
}

std::vector<int> sw_cuda_o2m(const std::string& query, 
                             const std::vector<std::string>& targets, 
                             SWConfig config) {
    std::vector<int> results;
    size_t num_targets = targets.size();
    if (num_targets == 0) return results;

    results.resize(num_targets);
    int query_len = query.length();

    // 1. Device Allocation (Query - Read Only)
    char* d_query;
    cudaCheck(cudaMalloc((void**)&d_query, query_len * sizeof(char)));
    cudaCheck(cudaMemcpy(d_query, query.c_str(), query_len, cudaMemcpyHostToDevice));

    // 2. Analyze Targets for Buffer Sizing
    size_t max_target_len = 0;
    for (const auto& t : targets) max_target_len = std::max(max_target_len, t.length());
    max_target_len += 64; // Safety padding

    // 3. Resource Pool Setup
    const int N_STREAMS = 4;
    cudaStream_t streams[N_STREAMS];

    // Resources
    char* d_target[N_STREAMS];
    char* h_target_pinned[N_STREAMS]; // NEW: Pinned Staging Buffers
    
    score_t *d_horiz[N_STREAMS], *d_vert[N_STREAMS], *d_diag[N_STREAMS];
    int* d_max_score[N_STREAMS];
    int* h_max_pinned[N_STREAMS];

    // Sizes
    size_t sz_target = max_target_len * sizeof(char);
    size_t sz_horiz  = (max_target_len + 1) * sizeof(score_t);
    size_t sz_vert   = (query_len + 1) * sizeof(score_t);
    int max_grid_cols = (max_target_len + TILE_DIM - 1) / TILE_DIM;
    size_t sz_diag   = (max_grid_cols + 1) * sizeof(score_t);

    // Allocation Loop
    for (int i = 0; i < N_STREAMS; i++) {
        cudaCheck(cudaStreamCreate(&streams[i]));
        
        // Target Buffers: Host (Pinned) and Device
        cudaCheck(cudaMallocHost((void**)&h_target_pinned[i], sz_target)); // Pinned!
        cudaCheck(cudaMalloc((void**)&d_target[i], sz_target));

        // Working Buffers
        cudaCheck(cudaMalloc((void**)&d_horiz[i], sz_horiz));
        cudaCheck(cudaMalloc((void**)&d_vert[i], sz_vert));
        cudaCheck(cudaMalloc((void**)&d_diag[i], sz_diag));
        cudaCheck(cudaMalloc((void**)&d_max_score[i], sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&h_max_pinned[i], sizeof(int)));
    }

    // 4. Execution Loop
    for (size_t i = 0; i < num_targets; ++i) {
        int s = i % N_STREAMS; 

        // A. SYNC STREAM
        // This ensures GPU is done with h_target_pinned[s] from the PREVIOUS cycle
        cudaCheck(cudaStreamSynchronize(streams[s]));

        // B. COLLECT RESULT (from previous job on this stream)
        if (i >= N_STREAMS) {
            results[i - N_STREAMS] = *h_max_pinned[s];
        }

        // C. PREPARE NEW JOB
        int target_len = targets[i].length();

        // 1. CPU Copy to Pinned Memory (Fast, no Driver sync)
        // Since we synced above, this is safe to overwrite.
        memcpy(h_target_pinned[s], targets[i].c_str(), target_len);

        // 2. Async DMA Copy to GPU
        // The driver sees the source is Pinned, so it proceeds asynchronously.
        cudaCheck(cudaMemcpyAsync(d_target[s], h_target_pinned[s], target_len, cudaMemcpyHostToDevice, streams[s]));

        // Reset buffers
        cudaCheck(cudaMemsetAsync(d_vert[s], 0, (query_len + 1) * sizeof(score_t), streams[s]));
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

    // 5. Final Cleanup
    cudaCheck(cudaDeviceSynchronize());

    // Collect remaining results
    size_t start_cleanup = (num_targets >= N_STREAMS) ? (num_targets - N_STREAMS) : 0;
    for (size_t j = start_cleanup; j < num_targets; j++) {
        int s = j % N_STREAMS;
        results[j] = *h_max_pinned[s];
    }

    // 6. Free Resources
    cudaFree(d_query);
    cleanup_resources(streams, d_target, h_target_pinned, d_horiz, d_vert, d_diag, d_max_score, h_max_pinned, N_STREAMS);

    return results;
}
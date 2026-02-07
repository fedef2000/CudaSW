#ifndef SW_CUDA_H
#define SW_CUDA_H

#include <string>
#include <vector>
#include <cuda_runtime.h>

typedef int score_t; 
#define TILE_DIM 32

// Internal macro for error checking
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#ifdef __cplusplus
extern "C" {
#endif

// Structure to hold scoring parameters
struct SWConfig {
    int match_score;
    int mismatch_score;
    int gap_score;
};

/**
 * Computes the Smith-Waterman alignment score using CUDA.
 * * @param seq1      Pointer to the first sequence (host memory)
 * @param len1      Length of the first sequence
 * @param seq2      Pointer to the second sequence (host memory)
 * @param len2      Length of the second sequence
 * @param config    Scoring configuration (match, mismatch, gap)
 * @return          The maximum alignment score found
 */
int sw_cuda_tiled(const std::string& seq1, const std::string& seq2,
                SWConfig config);

/**
 * Computes Smith-Waterman using the Diagonal Wavefront approach.
 * This is the "Basic Version" (non-tiled).
 */
int sw_cuda_diagonal(const std::string& seq1, const std::string& seq2,
                SWConfig config);

// 3. CPU Version (Baseline)
int sw_cpu(const std::string& seq1, const std::string& seq2,
                SWConfig config);                      
 
// 4. GPU with streams
std::vector<int> sw_cuda_o2m(const std::string& query, 
                             const std::vector<std::string>& targets, 
                             SWConfig config);


// Kernel declaration to be used in streams
#ifdef __CUDACC__
__global__ void compute_tile_kernel(const char* __restrict__ seq1, const char* __restrict__ seq2,
                                    score_t* d_horiz, score_t* d_vert, score_t* d_diag,
                                    int m, int n, int tile_step, int min_bx,
                                    int match, int mismatch, int gap,
                                    int* d_max_score);
#endif

#ifdef __cplusplus
}
#endif

#endif // SW_CUDA_H
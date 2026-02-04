#ifndef SW_CUDA_H
#define SW_CUDA_H

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
int sw_cuda_tiled(const char* seq1, int len1, 
                const char* seq2, int len2, 
                SWConfig config);

/**
 * Computes Smith-Waterman using the Diagonal Wavefront approach.
 * This is the "Basic Version" (non-tiled).
 */
int sw_cuda_diagonal(const char* seq1, int len1, 
                         const char* seq2, int len2, 
                         SWConfig config);

// 3. CPU Version (Baseline)
int sw_cpu(const char* seq1, int len1, 
               const char* seq2, int len2, 
               SWConfig config);                        
 
#ifdef __cplusplus
}
#endif

#endif // SW_CUDA_H
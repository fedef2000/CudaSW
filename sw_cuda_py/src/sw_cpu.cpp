#include "sw_cuda.h"
#include <vector>
#include <algorithm>
#include <iostream>

// Pure CPU implementation adapted for the library
int sw_cpu(const std::string& seq1, const std::string& seq2,
                SWConfig config)  {
    int len1 = seq1.length();
    int len2 = seq2.length();
    // Use vectors for memory management (RAII)
    // Only 2 rows are needed: O(min(N,M)) space complexity
    std::vector<int> prev_row(len2 + 1, 0);
    std::vector<int> curr_row(len2 + 1, 0);

    int max_score = 0;

    // Outer loop: iterate over rows (seq1)
    for (int i = 1; i <= len1; i++) {
        char char_s1 = seq1[i - 1]; 

        // Inner loop: iterate over columns (seq2)
        for (int j = 1; j <= len2; j++) {
            // Calculate scores
            int is_match = (char_s1 == seq2[j - 1]);
            int score_match_val = is_match ? config.match_score : config.mismatch_score;

            int score_diag = prev_row[j - 1] + score_match_val;
            int score_up   = prev_row[j]     + config.gap_score;
            int score_left = curr_row[j - 1] + config.gap_score;

            // Local alignment: clamp to 0
            int current_score = 0; 
            current_score = std::max(current_score, score_diag);
            current_score = std::max(current_score, score_up);
            current_score = std::max(current_score, score_left);

            curr_row[j] = current_score;
            max_score = std::max(max_score, current_score);
        }
        
        // Swap rows: current becomes previous
        // std::swap is efficient (swaps internal pointers of vectors)
        std::swap(prev_row, curr_row);
        
        // Reset the new start of the row to 0
        curr_row[0] = 0; 
    }

    return max_score;
}
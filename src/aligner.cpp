#include "cusw/aligner.hpp"
#include <algorithm> // for std::max
#include <vector>
#include <iostream>

namespace cusw {

    Aligner::Aligner(int gpu_id) {
        // We aren't using the GPU yet, but we keep the interface ready
        std::cout << "[CPU Mode] Aligner initialized.\n";
    }

    // The public batch function (The "Loop")
    std::vector<int> Aligner::score_batch(
        const std::vector<std::string>& queries,
        const std::vector<std::string>& targets,
        int match, int mismatch, int gap_open, int gap_extend
    ) {
        std::vector<int> results;
        results.reserve(queries.size());

        // Simple CPU Loop: Process one pair after another
        // Later, the GPU will do this whole loop in parallel!
        for (size_t i = 0; i < queries.size(); ++i) {
            int score = solve_single_cpu(queries[i], targets[i], match, mismatch, gap_open, gap_extend);
            results.push_back(score);
        }

        return results;
    }

    // The core Smith-Waterman Logic (O(N*M))
    int Aligner::solve_single_cpu(
        const std::string& seqA,
        const std::string& seqB,
        int match, int mismatch, int gap_open, int gap_extend
    ) {
        int n = seqA.length();
        int m = seqB.length();

        // Edge case: empty strings
        if (n == 0 || m == 0) return 0;

        // DP Matrix
        // We use a flat vector to simulate a 2D matrix of size (n+1) x (m+1)
        // Access index: i * (m + 1) + j
        int cols = m + 1;
        std::vector<int> H((n + 1) * (m + 1), 0);

        int max_score = 0;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                // 1. Calculate Match/Mismatch Score
                int diagonal_score = H[(i - 1) * cols + (j - 1)];
                if (seqA[i - 1] == seqB[j - 1]) {
                    diagonal_score += match;
                } else {
                    diagonal_score += mismatch;
                }

                // 2. Calculate Gap Scores
                // (Simplified affine gap: here treating open/extend as simple linear for MVP clarity, 
                // typically Smith-Waterman uses just 'gap' or distinct matrices for affine. 
                // Let's implement standard linear gap for the first pass to ensure correctness).
                // If you strictly need Affine (Open vs Extend), let me know, it requires 3 matrices.
                // Assuming Linear Gap for this MVP:
                int gap_score = gap_open; // Using gap_open as the linear penalty
                
                int up_score   = H[(i - 1) * cols + j] + gap_score;
                int left_score = H[i * cols + (j - 1)] + gap_score;

                // 3. Take the Max
                int current_score = 0; // Local alignment allows 0
                current_score = std::max(current_score, diagonal_score);
                current_score = std::max(current_score, up_score);
                current_score = std::max(current_score, left_score);

                H[i * cols + j] = current_score;
                
                // Track global max
                if (current_score > max_score) {
                    max_score = current_score;
                }
            }
        }

        return max_score;
    }
}

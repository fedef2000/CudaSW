#pragma once
#include <vector>
#include <string>

namespace cusw {

    class Aligner {
    public:
        // Constructor (we still keep gpu_id for later)
        Aligner(int gpu_id);

        /**
         * Main function to score a batch of sequences.
         * Currently runs on CPU.
         */
        std::vector<int> score_batch(
            const std::vector<std::string>& queries,
            const std::vector<std::string>& targets,
            int match_score,
            int mismatch_score,
            int gap_open,
            int gap_extend
        );

    private:
        // A private helper to calculate score for just ONE pair
        int solve_single_cpu(
            const std::string& query,
            const std::string& target,
            int match, int mismatch, int gap_open, int gap_extend
        );
    };

}

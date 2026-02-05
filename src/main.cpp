#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>   
#include <iomanip>
#include "sw_cuda.h" 

// Helper to print stats cleanly
void print_stats(const std::string& label, int score, double seconds, long long total_cells) {
    double gcups = (total_cells / 1e9) / seconds;
    
    std::cout << "--- " << label << " ---" << std::endl;
    std::cout << "Max Score:    " << score << std::endl;
    std::cout << "Time Elapsed: " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
    std::cout << "Performance:  " << std::fixed << std::setprecision(2) << gcups << " GCUPS" << std::endl;
    std::cout << std::endl;
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
        long long len1 = seq1.length();
        long long len2 = seq2.length();
        long long total_cells = len1 * len2;

        std::cout << "Processing Matrix: " << len1 << " x " << len2 << std::endl;
        std::cout << "Total Cells:       " << total_cells << std::endl;
        std::cout << "------------------------------------------------\n" << std::endl;

        // Setup configuration
        SWConfig config;
        config.match_score = 2;
        config.mismatch_score = -1;
        config.gap_score = -2;

        // --- 1. CPU Version (Conditional) ---
        int score_cpu = 0;
        bool ran_cpu = false;

        // Skip CPU if either sequence is longer than 500,000
        if (len1 <= 5000000 && len2 <= 5000000) {
            auto start = std::chrono::high_resolution_clock::now();
            
            score_cpu = sw_cpu(seq1, seq2, config);
            
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            
            print_stats("CPU Version", score_cpu, elapsed, total_cells);
            ran_cpu = true;
        } else {
            std::cout << "--- CPU Version ---" << std::endl;
            std::cout << "SKIPPED (Sequence > 500,000 chars)" << std::endl << std::endl;
        }

        // --- 2. GPU Basic (Diagonal) Version ---
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            int score_basic = sw_cuda_diagonal(seq1, seq2, config);
            
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            
            print_stats("GPU Basic (Diagonal)", score_basic, elapsed, total_cells);

            // Validation
            if (ran_cpu && score_basic != score_cpu) {
                std::cerr << "WARNING: Basic GPU score mismatch! (GPU: " << score_basic << ", CPU: " << score_cpu << ")" << std::endl;
            }
        }

        // --- 3. GPU Optimized (Tiled) Version ---
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Assuming sw_cuda_run is the tiled version in your library
            int score_tiled = sw_cuda_tiled(seq1, seq2, config);
            
            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            
            print_stats("GPU Optimized (Tiled)", score_tiled, elapsed, total_cells);

            // Validation
            if (ran_cpu && score_tiled != score_cpu) {
                std::cerr << "WARNING: Tiled GPU score mismatch! (GPU: " << score_tiled << ", CPU: " << score_cpu << ")" << std::endl;
            }
        }
    } else {
        std::cerr << "Error: Empty files." << std::endl;
    }
    return 0;
}
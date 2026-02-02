#include <string>
#include <vector>
#include <iostream>
#include <algorithm> 
#include <fstream>
#include <chrono>  
#include <iomanip>
#include <sys/resource.h>

const int match = 2;
const int mismatch = -1;
const int gap_penalty = -2;

void cpu_sw_benchmark(const std::string& s1, const std::string& s2) {
    long long n = s1.length(); // rows
    long long m = s2.length(); // columns
    long long total_cells = n * m;
    std::cout << "Starting processing..." << std::endl;
    std::cout << "Matrix Dimensions: " << n << " x " << m << std::endl;
    std::cout << "Total Cells: " << total_cells << std::endl;

    // Optimized Allocation: Only 2 rows instead of N rows, allocating the whole matrix is not possible if input sequences are very long
    std::vector<int> prev_row(m + 1, 0);
    std::vector<int> curr_row(m + 1, 0);

    int max_score = -1;

    // START TIMER
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. FILL STEP
    for (int i = 1; i <= n; i++) {
        // Optimization: pre-fetch character from s1 for the current row
        char char_s1 = s1[i - 1]; 

        for (int j = 1; j <= m; j++) {
            int score_diag = prev_row[j - 1] + ((char_s1 == s2[j - 1]) ? match : mismatch);
            int score_up   = prev_row[j] + gap_penalty;
            int score_left = curr_row[j - 1] + gap_penalty;

            int current_score = 0; // Local alignment clamps to 0

            if (score_diag > current_score) current_score = score_diag;
            if (score_up > current_score)   current_score = score_up;
            if (score_left > current_score) current_score = score_left;

            curr_row[j] = current_score;

            if (current_score > max_score) {
                max_score = current_score;
            }
        }
        // Swap rows: current becomes previous for the next iteration
        std::swap(prev_row, curr_row);
        
        // Reset column 0 for the next row (local alignment starts at 0)
        curr_row[0] = 0; 
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    double seconds = elapsed.count();
    double gcups = (total_cells / 1e9) / seconds;

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Peak Memory: " << usage.ru_maxrss << " KB" << std::endl;
    std::cout << "Max Score: " << max_score << std::endl;
    std::cout << "Time Elapsed: " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
    std::cout << "Performance: "  << std::fixed << std::setprecision(4) << gcups << " GCUPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
}

int main() {
    std::ifstream inFileA("a.seq");
    std::ifstream inFileB("b.seq");
    
    if (!inFileA || !inFileB) {
        std::cerr << "Error: Could not open one of the input files!" << std::endl;
        return 1;
    }

    std::string seq1, seq2;

    if (std::getline(inFileA, seq1) && std::getline(inFileB, seq2)) {
        cpu_sw_benchmark(seq1, seq2);
    } else {
        std::cerr << "Error: Empty files or invalid formatting." << std::endl;
    }
    return 0;
}
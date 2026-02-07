#include <iostream>
#include <string>
#include <vector>
#include <chrono>   
#include <iomanip>
#include <random>
#include "sw_cuda.h" 

// --- Helper: Generate Random DNA ---
std::string generate_random_sequence(size_t length) {
    const char bases[] = "ACGT";
    std::string s;
    s.reserve(length);
    
    // Use static random engine for reproducibility
    static std::mt19937 rng(42); 
    static std::uniform_int_distribution<int> dist(0, 3);

    for (size_t i = 0; i < length; ++i) {
        s += bases[dist(rng)];
    }
    return s;
}

// --- Helper: Print Stats ---
void print_stats(const std::string& label, double seconds, long long total_cells) {
    double gcups = (total_cells / 1e9) / seconds;
    
    std::cout << "--- " << label << " ---" << std::endl;
    std::cout << "Time Elapsed: " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
    std::cout << "Performance:  " << std::fixed << std::setprecision(2) << gcups << " GCUPS" << std::endl;
    std::cout << std::endl;
}

int main() {
    // --- 1. CONFIGURATION ---
    const int NUM_TARGETS = 10;
    const int LEN_QUERY   = 1000000;
    const int LEN_TARGET  = 1000;
    
    // Total cells = 10 * 10,000 * 1,000 = 100,000,000 cells
    long long total_cells = (long long)NUM_TARGETS * LEN_QUERY * LEN_TARGET;

    SWConfig config;
    config.match_score = 2;
    config.mismatch_score = -1;
    config.gap_score = -2;

    std::cout << "Generating Data..." << std::endl;
    std::cout << "Query:   1 sequence  x " << LEN_QUERY << " chars" << std::endl;
    std::cout << "Targets: " << NUM_TARGETS << " sequences x " << LEN_TARGET << " chars" << std::endl;
    std::cout << "Total Workload: " << total_cells << " cells" << std::endl;
    std::cout << "------------------------------------------------\n" << std::endl;

    // Generate Data
    std::string query = generate_random_sequence(LEN_QUERY);
    std::vector<std::string> targets;
    for(int i=0; i<NUM_TARGETS; i++) {
        targets.push_back(generate_random_sequence(LEN_TARGET));
    }

    // Storage for results to verify correctness
    std::vector<int> results_cpu(NUM_TARGETS);
    std::vector<int> results_basic(NUM_TARGETS);
    std::vector<int> results_tiled(NUM_TARGETS);
    std::vector<int> results_batch; // Will be resized by function

    // --- 2. CPU VERSION (Baseline) ---
    {
        std::cout << "Running CPU Loop..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_TARGETS; ++i) {
            results_cpu[i] = sw_cpu(query, 
                                        targets[i], 
                                        config);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        print_stats("CPU Version (Looped)", elapsed, total_cells);
    }

    // --- 3. GPU BASIC (Diagonal) ---
    {
        std::cout << "Running GPU Basic Loop..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_TARGETS; ++i) {
            results_basic[i] = sw_cuda_diagonal(query.c_str(),
                                                    targets[i],
                                                    config);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        print_stats("GPU Basic (Diagonal, Looped)", elapsed, total_cells);

        // Verify
        bool ok = true;
        for(int i=0; i<NUM_TARGETS; i++) if(results_basic[i] != results_cpu[i]) ok = false;
        if(!ok) std::cerr << "ERROR: GPU Basic results do not match CPU!\n";
    }

    // --- 4. GPU TILED (Standard Optimized) ---
    {
        std::cout << "Running GPU Tiled Loop..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_TARGETS; ++i) {
            // Assuming sw_cuda_run is your standard tiled single-pair function
            results_tiled[i] = sw_cuda_tiled(query.c_str(),
                                           targets[i],
                                           config);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        print_stats("GPU Tiled (Standard, Looped)", elapsed, total_cells);

        // Verify
        bool ok = true;
        for(int i=0; i<NUM_TARGETS; i++) if(results_tiled[i] != results_cpu[i]) ok = false;
        if(!ok) std::cerr << "ERROR: GPU Tiled results do not match CPU!\n";
    }

    // --- 5. GPU BATCHED (One-to-Many Optimized) ---
    {
        std::cout << "Running GPU Batched (Streamed)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        // This function handles the entire vector of targets at once using streams
        results_batch = sw_cuda_o2m(query, targets, config);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        print_stats("GPU Batched (One-to-Many)", elapsed, total_cells);

        // Verify & Print Scores
        bool ok = true;
        std::cout << "Results Verification:" << std::endl;
        for(int i=0; i<NUM_TARGETS; i++) {
            if (results_batch[i] != results_cpu[i]) {
                ok = false;
                std::cout << "  Seq " << i << ": FAIL (CPU: " << results_cpu[i] 
                          << ", Batch: " << results_batch[i] << ")" << std::endl;
            } else {
                 // Print first few scores to show it worked
                 if (i < 3) std::cout << "  Seq " << i << ": OK (Score: " << results_batch[i] << ")" << std::endl;
            }
        }
        if (ok) std::cout << "  [SUCCESS] All scores match." << std::endl;
    }

    return 0;
}
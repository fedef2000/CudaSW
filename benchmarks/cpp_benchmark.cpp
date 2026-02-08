#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <map>

// Include your library header
// We assume this file is in 'benchmarks/', so we go up one level to find the header
#include "../sw_cuda_py/src/sw_cuda.h"

namespace fs = std::filesystem;

// --- Configuration ---
const std::string DATA_DIR = "./data";
const int MATCH_SCORE = 2;
const int MISMATCH_SCORE = -2;
const int GAP_SCORE = -1;

struct Sequence {
    std::string name;
    std::string content;
};

// --- Helper Functions ---

std::vector<Sequence> load_sequences(const std::string& folder_name) {
    std::vector<Sequence> seqs;
    std::string full_path = DATA_DIR + "/" + folder_name;
    
    if (!fs::exists(full_path)) {
        std::cerr << "[Error] Directory " << full_path << " not found!" << std::endl;
        return seqs;
    }

    for (const auto& entry : fs::directory_iterator(full_path)) {
        std::string ext = entry.path().extension().string();
        // Check for common FASTA extensions
        if (ext == ".fasta" || ext == ".fa" || ext == ".seq") {
            std::ifstream file(entry.path());
            std::string line;
            std::string current_header;
            std::string current_seq;

            while (std::getline(file, line)) {
                // Skip empty lines
                if (line.empty()) continue;
                // Remove carriage return if present (Windows format compatibility)
                if (line.back() == '\r') line.pop_back();

                if (line[0] == '>') {
                    // If we have accumulating data for a previous sequence, save it now
                    if (!current_header.empty()) {
                        seqs.push_back({current_header, current_seq});
                    }
                    
                    // Start new sequence
                    current_header = line.substr(1); // Remove '>'
                    current_seq.clear();
                } else {
                    // It's sequence data
                    current_seq += line;
                }
            }

            // Save the last sequence in the file
            if (!current_header.empty()) {
                seqs.push_back({current_header, current_seq});
            } 
            // Fallback: If file has content but no '>' headers (raw sequence file)
            else if (!current_seq.empty()) {
                seqs.push_back({entry.path().filename().string(), current_seq});
            }
        }
    }
    std::cout << "Loaded " << seqs.size() << " sequences from " << folder_name << std::endl;
    return seqs;
}

double calculate_gcups(unsigned long long total_cells, double elapsed_seconds) {
    if (elapsed_seconds <= 0) return 0.0;
    return (total_cells / 1e9) / elapsed_seconds;
}

// Type definition for the pairwise functions (CPU, Tiled, Diagonal)
typedef int (*AlgoFunc)(const std::string&, const std::string&, SWConfig);

// --- Test Runner ---

std::map<std::pair<std::string, std::string>, int> run_pairwise_test(
    std::string label, 
    AlgoFunc func, 
    const std::vector<Sequence>& queries, 
    const std::vector<Sequence>& targets, 
    SWConfig config,
    const std::map<std::pair<std::string, std::string>, int>* ref_results = nullptr) 
{
    std::cout << "\n--- Testing: " << label << " ---" << std::endl;

    // Warmup (for GPU functions)
    if (label.find("CUDA") != std::string::npos && !queries.empty() && !targets.empty()) {
        func(queries[0].content, targets[0].content, config);
    }

    std::map<std::pair<std::string, std::string>, int> results;
    unsigned long long total_cells = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& q : queries) {
        for (const auto& t : targets) {
            int score = func(q.content, t.content, config);
            
            // Store result
            results[{q.name, t.name}] = score;
            total_cells += (unsigned long long)q.content.length() * t.content.length();

            // Verify
            if (ref_results) {
                auto it = ref_results->find({q.name, t.name});
                if (it != ref_results->end() && it->second != score) {
                    std::cerr << "  [ERROR] Mismatch " << q.name << " vs " << t.name 
                              << ": Got " << score << ", Expected " << it->second << std::endl;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double gcups = calculate_gcups(total_cells, elapsed.count());
    std::cout << "  Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "  Total Cells: " << total_cells << std::endl;
    std::cout << "  Performance: " << gcups << " GCUPS" << std::endl;

    return results;
}

int main() {
    // 1. Setup
    SWConfig config = {MATCH_SCORE, MISMATCH_SCORE, GAP_SCORE};
    std::cout << "--- C++ CUDA Smith-Waterman Benchmark ---" << std::endl;

    // 2. Load Data
    auto queries = load_sequences("query");
    auto targets = load_sequences("target");

    if (queries.empty() || targets.empty()) {
        std::cerr << "Aborting: No data found." << std::endl;
        return 1;
    }

    // 3. CPU Baseline
    auto cpu_results = run_pairwise_test("CPU Baseline", sw_cpu, queries, targets, config);

    // 4. GPU Diagonal
    run_pairwise_test("CUDA Diagonal", sw_cuda_diagonal, queries, targets, config, &cpu_results);

    // 5. GPU Tiled
    run_pairwise_test("CUDA Tiled", sw_cuda_tiled, queries, targets, config, &cpu_results);

    // 6. GPU Batch (One-to-Many)
    std::cout << "\n--- Testing: CUDA Batch (O2M) ---" << std::endl;
    
    // Prepare data for batch interface
    // Note: This takes the first query and runs it against all targets
    std::string q_seq = queries[0].content;
    std::vector<std::string> target_list;
    unsigned long long batch_cells = 0;
    
    for(const auto& t : targets) {
        target_list.push_back(t.content);
        batch_cells += (unsigned long long)q_seq.length() * t.content.length();
    }

    // Warmup
    sw_cuda_o2m(q_seq, target_list, config);

    auto start = std::chrono::high_resolution_clock::now();
    
    // Call the batch function
    std::vector<int> batch_scores = sw_cuda_o2m(q_seq, target_list, config);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double gcups = calculate_gcups(batch_cells, elapsed.count());
    std::cout << "  Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "  Performance: " << gcups << " GCUPS (Fastest!)" << std::endl;

    return 0;
}
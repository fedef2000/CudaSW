#include "cusw/aligner.hpp"
#include "kernels.cuh"
#include <iostream>

namespace cusw {
    Aligner::Aligner(int gpu_id) {
        std::cout << "Initializing Aligner on GPU " << gpu_id << "\n";
    }

    std::vector<int> Aligner::score_batch(const std::vector<std::string>& a, 
                                          const std::vector<std::string>& b) {
        // Trigger the GPU kernel just to prove it works
        launch_test_kernel();
        
        // Return dummy data
        std::vector<int> results;
        for(size_t i=0; i < a.size(); ++i) results.push_back(42);
        return results;
    }
}

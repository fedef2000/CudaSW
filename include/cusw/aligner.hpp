#pragma once
#include <string>
#include <vector>

namespace cusw {
    // A simple class to test the binding
    class Aligner {
    public:
        Aligner(int gpu_id);
        std::vector<int> score_batch(const std::vector<std::string>& a, 
                                     const std::vector<std::string>& b);
    };
}

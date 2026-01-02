#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <cerrno>
#include <climits>

const char alphabet_bases[] = {'A', 'C', 'G', 'T'};
const int bases_len = 4;

void write_file(const std::string& filename, int length, std::mt19937& gen) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not create " << filename << std::endl;
        return;
    }

    std::uniform_int_distribution<> dis(0, bases_len - 1);

    for (int i = 0; i < length; i++) {
        file << alphabet_bases[dis(gen)];
    }
    
    file.close();
    std::cout << "Generated " << filename << " with length " << length << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <length1> <length2> ... <lengthN>" << std::endl;
        return 1;
    }

    int num_files = argc - 1;
    if (num_files > 26) {
        std::cerr << "Error: Maximum 26 files (a-z) supported." << std::endl;
        return 1;
    }

    // Initialize modern random engine once
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < num_files; i++) {
        char *p;
        errno = 0;
        long conv = std::strtol(argv[i + 1], &p, 10);

        if (errno != 0 || *p != '\0' || conv > INT_MAX || conv < 0) {
            std::cerr << "Error: Argument " << i + 1 << " (" << argv[i + 1] << ") is not a valid positive integer." << std::endl;
            continue;
        }

        int length = static_cast<int>(conv);
        
        // Generate filename: 0 -> 'a.seq', 1 -> 'b.seq', etc.
        std::string filename = "";
        filename += (char)('a' + i);
        filename += ".seq";

        write_file(filename, length, gen);
    }

    return 0;
}
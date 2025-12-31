#include <iostream>
#include <cstdlib>   // per strtol ed exit
#include <cerrno>    // per errno
#include <climits>   // per INT_MAX, INT_MIN
#include <fstream>

const int DEFAULT_VAL = 1000;
const char alphabet[] = {'A', 'C', 'G', 'T'};
const int alphabet_len = 4;

void write_file(char * filename, int length){
    std::ofstream file(filename);

    for(int i =0;i<length; i++){
        int r = rand()%alphabet_len;
        char c = alphabet[r];
        file << c;
    }
}

int main(int argc, char** argv) {
    int length = 0;

    if (argc < 2) {
        std::cout << "No length inserted, default value is " << DEFAULT_VAL << std::endl;
        length = DEFAULT_VAL;
    } else {
        char *p;
        errno = 0;
        long conv = strtol(argv[1], &p, 10);

        // Controllo errori di conversione
        if (errno != 0 || *p != '\0' || conv > INT_MAX || conv < INT_MIN) {
            std::cerr << "The inserted value is not valid" << std::endl;
            std::cerr << "Correct use is: " << argv[0] << " <length>" << std::endl;
            return 1; // Termina con errore
        }
        length = static_cast<int>(conv);
    }

    write_file("a.seq", length);
    return 0;
}
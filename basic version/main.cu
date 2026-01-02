#include <string>
#include <vector>
#include <iostream>
#include <algorithm> 
#include <ifstream>

const int match = 2;
const int mismatch = -1;
const int gap_penalty = -2;

enum directions { STOP, UP, LEFT, DIAGONAL };

int cpu_sw(std::string s1, std::string s2) {
    int n = s1.length(); // rows
    int m = s2.length(); // columns
    int max_score = -1;
    int max_index = 0;
    std::vector<int> H((n + 1) * (m + 1), 0);
    std::vector<directions> backtrace((n + 1) * (m + 1), STOP);

    // 1. FILL STEP
    for (int row = 1; row < n + 1; row++) {
        for (int col = 1; col < m + 1; col++) {
            int current = col + (m + 1) * row;
            int prev_diag = (col - 1) + (m + 1) * (row - 1);
            int prev_top = col + (m + 1) * (row - 1);
            int prev_left = (col - 1) + (m + 1) * row;

            int score_diag = H[prev_diag] + ((s1[row - 1] == s2[col - 1]) ? match : mismatch);
            int score_up = H[prev_top] + gap_penalty;
            int score_left = H[prev_left] + gap_penalty;

            int current_score = 0;
            directions d = STOP;

            if (score_diag >= current_score) {
                current_score = score_diag;
                d = DIAGONAL;
            }
            if (score_up > current_score) {
                current_score = score_up;
                d = UP;
            }
            if (score_left > current_score) {
                current_score = score_left;
                d = LEFT;
            }

            H[current] = current_score;
            backtrace[current] = d;

            if (current_score > max_score) {
                max_score = current_score;
                max_index = current;
            }
        }
    }
    std::cout << "Max Score: " << max_score << std::endl;

    // 2. TRACEBACK STEP
    std::string align1 = "";
    std::string align2 = "";

    int curr_idx = max_index;
    
    int curr_row = curr_idx / (m + 1);
    int curr_col = curr_idx % (m + 1);

    while (backtrace[curr_idx] != STOP && curr_idx != 0) {
        directions d = backtrace[curr_idx];

        switch (d) {
            case DIAGONAL:
                // Match/Mismatch: prendiamo caratteri da entrambe
                align1 += s1[curr_row - 1];
                align2 += s2[curr_col - 1];
                curr_row--;
                curr_col--;
                break; 

            case UP:
                // Gap in S2 (muovo solo riga)
                align1 += s1[curr_row - 1];
                align2 += '-';
                curr_row--;
                break; 

            case LEFT:
                // Gap in S1 (muovo solo colonna)
                align1 += '-';
                align2 += s2[curr_col - 1];
                curr_col--;
                break; 
                
            case STOP:
                break;
        }
        curr_idx = curr_col + (m + 1) * curr_row;
    }

    std::reverse(align1.begin(), align1.end());
    std::reverse(align2.begin(), align2.end());

    std::cout << "Alignment:" << std::endl;
    std::cout << align1 << std::endl;
    std::cout << align2 << std::endl;

    return max_score;
}

int main() {
    std::ifstream inFileA("a.seq");
    std::ifstream inFileB("b.seq");
    if (!inFileA || !inFileB) {
        std::cerr << "Errore: Impossibile aprire uno dei file di input!" << std::endl;
        return 1;
    }

    std::string seq1, seq2;

    if (std::getline(inFileA, seq1) && std::getline(inFileB, seq2)) {
        cpu_sw(seq1, seq2);
    } else {
        std::cerr << "Errore: File vuoti o formattazione non valida." << std::endl;
    }
    return 0;
}
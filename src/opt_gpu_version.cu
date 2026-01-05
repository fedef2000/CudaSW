#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <chrono> 
#include <iomanip>

// --- CONFIGURAZIONE ---
#define MATCH 2
#define MISMATCH -1
#define GAP -2
#define BLOCK_SIZE 256

// Usiamo short invece di int per risparmiare il 50% di banda di memoria
typedef int score_t; 

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)

/**
 * KERNEL OTTIMIZZATO CON SHARED MEMORY
 * * 1. Carica segmenti di seq1 e seq2 in Shared Memory (Tiling dei dati)
 * 2. Esegue reduction locale per il max_score (meno atomiche)
 */
__global__ void compute_diagonal_tiled(score_t* d_cur, score_t* d_prev, score_t* d_prev2, 
                                       const char* __restrict__ seq1, const char* __restrict__ seq2, 
                                       int m, int n, int k, 
                                       int min_r, int diag_len,
                                       int* d_max_score) {
    
    // --- 1. SHARED MEMORY ALLOCATION ---
    // Ogni thread ha bisogno di 1 char da seq1 e 1 char da seq2.
    // Carichiamo un blocco di caratteri per tutto il thread block.
    __shared__ char s_seq1[BLOCK_SIZE];
    __shared__ char s_seq2[BLOCK_SIZE];
    
    // Buffer per la reduction del punteggio massimo
    __shared__ int s_max_score[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = tid + blockIdx.x * blockDim.x; // Global ID sulla diagonale

    // Inizializza score locale a 0
    int my_score = 0; 

    // --- 2. CARICAMENTO DATI IN SHARED MEMORY (Cooperative Loading) ---
    // Calcoliamo le coordinate (r, c) per questo thread
    int r = -1, c = -1;
    bool active = gid < diag_len;

    if (active) {
        r = min_r + gid;       
        c = (k + 2) - r;
        
        // Carichiamo i caratteri corrispondenti in Shared Memory
        // Nota: seq1 e seq2 sono 0-indexed, quindi r-1 e c-1
        // Usiamo __ldg() se disponibile (automatico su architetture moderne con const pointer)
        s_seq1[tid] = seq1[r - 1];
        s_seq2[tid] = seq2[c - 1];
    } else {
        // Padding per thread inattivi (evita letture sporche)
        s_seq1[tid] = 0;
        s_seq2[tid] = 0;
    }

    // Barriera: aspettiamo che tutti abbiano caricato i dati
    __syncthreads();

    // --- 3. CALCOLO DELLO SCORE ---
    if (active) {
        // Calcolo indici buffer precedenti (logica identica alla versione base)
        int min_r_prev = (k + 1 < n) ? 1 : (k + 1) - n; 
        int min_r_prev2 = (k < n) ? 1 : k - n;

        int idx_up   = (r - 1) - min_r_prev; 
        int idx_left = r - min_r_prev;       
        int idx_diag = (r - 1) - min_r_prev2; 

        score_t val_up = 0;
        score_t val_left = 0;
        score_t val_diag = 0;

        // Lettura dalla Global Memory per i punteggi precedenti
        // (Nota: ottimizzare anche questo richiederebbe un tiling 2D più complesso)
        if (r > 1 && c > 1) val_diag = d_prev2[idx_diag];
        if (r > 1) val_up = d_prev[idx_up];
        if (c > 1) val_left = d_prev[idx_left];

        // Lettura dalle sequenze ora avviene da SHARED MEMORY (velocissima)
        char b1 = s_seq1[tid];
        char b2 = s_seq2[tid];

        int match_score = (b1 == b2) ? MATCH : MISMATCH;
        
        int s_diag = (int)val_diag + match_score;
        int s_up   = (int)val_up + GAP;
        int s_left = (int)val_left + GAP;

        my_score = max(0, s_diag);
        my_score = max(my_score, s_up);
        my_score = max(my_score, s_left);

        d_cur[gid] = (score_t)my_score;
    }

    // --- 4. REDUCTION DEL MAX SCORE (Ottimizzazione Atomica) ---
    
    // Carica il proprio score nel buffer condiviso
    s_max_score[tid] = my_score;
    __syncthreads();

    // Reduction ad albero in Shared Memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_max_score[tid] = max(s_max_score[tid], s_max_score[tid + s]);
        }
        __syncthreads();
    }

    // Solo il thread 0 del blocco aggiorna la memoria globale
    if (tid == 0) {
        int block_max = s_max_score[0];
        if (block_max > 0) {
            atomicMax(d_max_score, block_max);
        }
    }
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
        long long m = seq1.size(); // rows
        long long n = seq2.size(); // cols
        long long total_cells = m * n;

        // --- Sequence Allocation (Device) ---
        char *d_seq1, *d_seq2;
        cudaMalloc((void **)&d_seq1, m * sizeof(char));
        cudaMalloc((void **)&d_seq2, n * sizeof(char));
        cudaMemcpy(d_seq1, seq1.data(), m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_seq2, seq2.data(), n, cudaMemcpyHostToDevice);

        // --- Diagonal Buffer Allocation (Device) ---
        // OTTIMIZZAZIONE: Usiamo score_t (short) invece di int
        score_t *d_current, *d_prev, *d_prev2;
        size_t diag_len = std::min(m, n);
        size_t diag_bytes = diag_len * sizeof(score_t); 

        cudaMalloc((void **)&d_current, diag_bytes);
        cudaMalloc((void **)&d_prev, diag_bytes);
        cudaMalloc((void **)&d_prev2, diag_bytes);
        
        cudaMemset(d_current, 0, diag_bytes);
        cudaMemset(d_prev, 0, diag_bytes);
        cudaMemset(d_prev2, 0, diag_bytes);

        // --- Max Score Variable ---
        int* d_max_score;
        cudaMalloc((void**)&d_max_score, sizeof(int));
        cudaMemset(d_max_score, 0, sizeof(int));

        const int total_diagonals = m + n - 1;

        std::cout << "Starting computation on " << m << " x " << n << " matrix..." << std::endl;

        // --- START TIMER ---
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int k = 0; k < total_diagonals; k++) {
            int min_r = std::max(1LL, k + 2 - n);
            int max_r = std::min((long long)m, (long long)k + 1);
            int diagonal_dimension = max_r - min_r + 1;
            
            if (diagonal_dimension <= 0) continue;

            int gridSize = (diagonal_dimension + BLOCK_SIZE - 1) / BLOCK_SIZE;

            compute_diagonal_tiled<<<gridSize, BLOCK_SIZE>>>(d_current, d_prev, d_prev2, 
                                                      d_seq1, d_seq2, 
                                                      m, n, k, 
                                                      min_r, diagonal_dimension,
                                                      d_max_score);
            
            // Buffer swap (solo puntatori)
            score_t* temp = d_prev2;
            d_prev2 = d_prev;
            d_prev = d_current;
            d_current = temp;
        }
        
        cudaDeviceSynchronize();
        cudaCheckErrors("Execution failed");

        // --- STOP TIMER ---
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        int h_max_score = 0;
        cudaMemcpy(&h_max_score, d_max_score, sizeof(int), cudaMemcpyDeviceToHost);

        double seconds = elapsed.count();
        double gcups = (total_cells / 1e9) / seconds;

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Optimization:    Shared Mem Tiling + Reduction + Short Type" << std::endl;
        std::cout << "Max Score Found: " << h_max_score << std::endl;
        std::cout << "Time Elapsed:    " << std::fixed << std::setprecision(4) << seconds << " s" << std::endl;
        std::cout << "Performance:     " << std::fixed << std::setprecision(4) << gcups << " GCUPS" << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        cudaFree(d_seq1); cudaFree(d_seq2);
        cudaFree(d_current); cudaFree(d_prev); cudaFree(d_prev2);
        cudaFree(d_max_score);
    } else {
        std::cerr << "Error: Empty files." << std::endl;
    }

    return 0;
}
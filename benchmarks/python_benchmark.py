import os
import glob
import time
import sw_cuda_py

# --- Check for Biopython ---
try:
    from Bio import Align
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("Warning: Biopython not found. Skipping Biopython benchmark.")

# --- Configuration ---
DATA_DIR = "./data"
MATCH = 3
MISMATCH = -3
GAP = -1

def load_sequences(folder_name):
    """
    Reads .seq, .fasta, and .fa files in a folder.
    Parses FASTA format (headers starting with >).
    Falls back to treating the whole file as a sequence if no headers are found.
    Returns a list of (name, content).
    """
    folder_path = os.path.join(DATA_DIR, folder_name)
    sequences = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Directory {folder_path} not found")
        return []

    # Support multiple extensions
    extensions = ['*.seq', '*.fasta', '*.fa']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        print(f"Warning: No sequence files found in {folder_path}")
        return []

    for f_path in files:
        with open(f_path, 'r') as file:
            lines = file.readlines()

        file_seqs = []
        current_header = None
        current_seq = []
        has_fasta_header = False

        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith(">"):
                has_fasta_header = True
                if current_header:
                    file_seqs.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header and current_seq:
            file_seqs.append((current_header, "".join(current_seq)))
        
        if not has_fasta_header and current_seq:
            name = os.path.basename(f_path)
            file_seqs.append((name, "".join(current_seq)))
            
        sequences.extend(file_seqs)
    
    print(f"Loaded {len(sequences)} sequences from {folder_name}")
    return sequences

def calculate_gcups(total_cells, elapsed_seconds):
    if elapsed_seconds <= 0: return 0.0
    return (total_cells / 1e9) / elapsed_seconds

# --- Biopython Wrapper ---
def biopython_sw(q, t, config):
    """
    Wrapper to run Biopython's PairwiseAligner with the given config.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'  # Smith-Waterman is local alignment
    
    # Map our config to Biopython's scoring
    aligner.match_score = config.match_score
    aligner.mismatch_score = config.mismatch_score
    
    # For linear gap penalty (simple gap score), open and extend are the same
    aligner.open_gap_score = config.gap_score
    aligner.extend_gap_score = config.gap_score
    
    # Calculate score
    return int(aligner.score(q, t))

def run_test(label, func, queries, targets, config, check_ref=None):
    print(f"\n--- Testing: {label} ---")
    
    if not queries or not targets:
        print("  [Skipping] Missing queries or targets.")
        return {}

    # Warmup
    if "cuda" in label.lower():
        try:
            func(queries[0][1], targets[0][1], config)
        except Exception as e:
            print(f"  [Warmup Failed] {e}")

    start_total = time.perf_counter()
    
    total_cells = 0
    results = {} 
    
    for q_name, q_seq in queries:
        for t_name, t_seq in targets:
            
            t_start = time.perf_counter()
            score = func(q_seq, t_seq, config)
            t_end = time.perf_counter()
            
            duration = t_end - t_start
            cells = len(q_seq) * len(t_seq)
            gcups = calculate_gcups(cells, duration)
            
            print(f"  [{q_name} vs {t_name}] Score: {score}, Time: {duration:.5f}s, Perf: {gcups:.4f} GCUPS")

            total_cells += cells
            results[(q_name, t_name)] = score

            if check_ref:
                ref_score = check_ref.get((q_name, t_name))
                # Note: Biopython might return float scores, cast to int for comparison if needed
                if ref_score is not None and int(score) != int(ref_score):
                    print(f"    >>> [ERROR] Mismatch! Expected {ref_score}, Got {score}")

    end_total = time.perf_counter()
    elapsed_total = end_total - start_total
    
    avg_gcups = calculate_gcups(total_cells, elapsed_total)
    
    print(f"  --- {label} Summary ---")
    print(f"  Total Time: {elapsed_total:.4f}s")
    print(f"  Average Performance: {avg_gcups:.4f} GCUPS")
    
    return results

def main():
    config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    print(f"Configuration: Match={MATCH}, Mismatch={MISMATCH}, Gap={GAP}")
    
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data.")
        return

    # 1. Baseline: Custom C++ CPU
    cpu_results = run_test("CPU Baseline (C++)", sw_cuda_py.sw_cpu, queries, targets, config)
    
    # 2. Baseline: Biopython (if available)
    if HAS_BIOPYTHON:
        run_test("Biopython (CPU)", biopython_sw, queries, targets, config, check_ref=cpu_results)

    # 3. GPU: Diagonal
    run_test("CUDA Diagonal", sw_cuda_py.sw_cuda_diagonal, queries, targets, config, check_ref=cpu_results)
    
    # 4. GPU: Tiled
    run_test("CUDA Tiled", sw_cuda_py.sw_cuda_tiled, queries, targets, config, check_ref=cpu_results)

    # 5. GPU: Batch (Streams)
    print(f"\n--- Testing: CUDA Batch (O2M) ---")
    if len(queries) > 0 and len(targets) > 0:
        
        # Prepare target list (shared across all queries)
        t_list = [t[1] for t in targets]
        
        # Warmup with the first query
        try:
            sw_cuda_py.sw_cuda_o2m(queries[0][1], t_list, config)
        except Exception as e:
            print(f"  [Warmup Failed] {e}")

        total_batch_time = 0.0
        total_batch_cells = 0

        # Iterate over ALL queries
        for q_name, q_seq in queries:
            
            # Calculate total matrix cells for this specific 1-to-many batch
            current_batch_cells = len(q_seq) * sum(len(t) for t in t_list)

            start = time.perf_counter()
            # Run the O2M kernel for the current query against all targets
            batch_scores = sw_cuda_py.sw_cuda_o2m(q_seq, t_list, config)
            end = time.perf_counter()
            
            duration = end - start
            gcups = calculate_gcups(current_batch_cells, duration)
            
            print(f"  [Batch: {q_name} vs {len(t_list)} targets] Time: {duration:.5f}s, Perf: {gcups:.4f} GCUPS")
            
            total_batch_time += duration
            total_batch_cells += current_batch_cells

            # Optional: Check accuracy if we have CPU reference results
            if cpu_results and batch_scores:
                for idx, t_tuple in enumerate(targets):
                    t_name = t_tuple[0]
                    ref_score = cpu_results.get((q_name, t_name))
                    
                    # Assuming sw_cuda_o2m returns a list of scores corresponding to t_list order
                    if ref_score is not None and idx < len(batch_scores):
                        if int(batch_scores[idx]) != int(ref_score):
                             print(f"    >>> [ERROR] Mismatch! {q_name} vs {t_name}. Expected {ref_score}, Got {batch_scores[idx]}")

        # Summary for the batch phase
        avg_gcups = calculate_gcups(total_batch_cells, total_batch_time)
        print(f"  --- CUDA Batch (O2M) Summary ---")
        print(f"  Total Time: {total_batch_time:.4f}s")
        print(f"  Average Performance: {avg_gcups:.4f} GCUPS")

if __name__ == "__main__":
    main()
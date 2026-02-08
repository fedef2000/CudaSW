import os
import glob
import time
import sw_cuda_py

# --- Configuration ---
DATA_DIR = "./data"  # Assumes script is in 'benchmarks/' and data is in 'root/data'
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
                # If we were building a sequence, save it
                if current_header:
                    file_seqs.append((current_header, "".join(current_seq)))
                
                # Start new sequence
                current_header = line[1:] # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Save the last sequence found in the file
        if current_header and current_seq:
            file_seqs.append((current_header, "".join(current_seq)))
        
        # Fallback: If file had content but no '>' headers, treat as raw sequence
        if not has_fasta_header and current_seq:
            # Use filename as the sequence name
            name = os.path.basename(f_path)
            file_seqs.append((name, "".join(current_seq)))
            
        sequences.extend(file_seqs)
    
    print(f"Loaded {len(sequences)} sequences from {folder_name}")
    return sequences

def calculate_gcups(total_cells, elapsed_seconds):
    if elapsed_seconds <= 0: return 0.0
    return (total_cells / 1e9) / elapsed_seconds

def run_test(label, func, queries, targets, config, check_ref=None):
    """
    Runs a benchmark for a specific function.
    Prints GCUPS for each individual function call.
    check_ref: If provided (dict), validates the score against this reference.
    """
    print(f"\n--- Testing: {label} ---")
    
    if not queries or not targets:
        print("  [Skipping] Missing queries or targets.")
        return {}

    # 1. Warmup (crucial for GPU to initialize context)
    if "cuda" in label.lower():
        try:
            # Run a dummy call to wake up the GPU
            func(queries[0][1], targets[0][1], config)
        except Exception as e:
            print(f"  [Warmup Failed] {e}")

    start_total = time.perf_counter()
    
    total_cells = 0
    results = {} # Store results for validation
    
    # Loop over every combination of Query vs Target
    for q_name, q_seq in queries:
        for t_name, t_seq in targets:
            
            # --- Per-Call Timing Start ---
            t_start = time.perf_counter()
            
            # Call the C++ function
            score = func(q_seq, t_seq, config)
            
            # --- Per-Call Timing End ---
            t_end = time.perf_counter()
            
            # Metrics for this specific call
            duration = t_end - t_start
            cells = len(q_seq) * len(t_seq)
            gcups = calculate_gcups(cells, duration)
            
            # Print immediate result
            print(f"  [{q_name} vs {t_name}] Cells: {cells/1000000}M, Score: {score}, Time: {duration:.5f}s, Perf: {gcups:.4f} GCUPS")

            # Accumulate totals
            total_cells += cells
            results[(q_name, t_name)] = score

            # Optional Validation
            if check_ref:
                ref_score = check_ref.get((q_name, t_name))
                if ref_score is not None and score != ref_score:
                    print(f"    >>> [ERROR] Mismatch! Expected {ref_score}, Got {score}")

    end_total = time.perf_counter()
    elapsed_total = end_total - start_total
    
    avg_gcups = calculate_gcups(total_cells, elapsed_total)
    
    print(f"  --- {label} Summary ---")
    print(f"  Total Time: {elapsed_total:.4f}s")
    print(f"  Average Performance: {avg_gcups:.4f} GCUPS")
    
    return results

def main():
    # 1. Setup
    config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    print(f"Configuration: Match={MATCH}, Mismatch={MISMATCH}, Gap={GAP}")
    
    # 2. Load Data
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data. Please ensure 'data/query' and 'data/target' exist and contain .seq or .fasta files.")
        return

    # 3. Baseline: CPU
    # Note: sw_cpu is slow, so if datasets are huge, you might want to limit this
    cpu_results = run_test("CPU Baseline", sw_cuda_py.sw_cpu, queries, targets, config)
    
    # 4. GPU: Diagonal Wavefront
    run_test("CUDA Diagonal", sw_cuda_py.sw_cuda_diagonal, queries, targets, config, check_ref=cpu_results)
    
    # 5. GPU: Tiled
    run_test("CUDA Tiled", sw_cuda_py.sw_cuda_tiled, queries, targets, config, check_ref=cpu_results)

    # 6. GPU: One-to-Many (Batch)
    print(f"\n--- Testing: CUDA Batch (O2M) ---")
    
    if len(queries) > 0 and len(targets) > 0:
        q_name, q_seq = queries[0] # Take first query
        t_list = [t[1] for t in targets] # List of all targets
        
        # Calculate expected cells for GCUPS
        batch_cells = len(q_seq) * sum(len(t) for t in t_list)

        # Warmup
        try:
            sw_cuda_py.sw_cuda_o2m(q_seq, t_list, config)
        except:
            pass

        # Single Batch Call
        start = time.perf_counter()
        scores = sw_cuda_py.sw_cuda_o2m(q_seq, t_list, config)
        end = time.perf_counter()
        
        duration = end - start
        gcups = calculate_gcups(batch_cells, duration)
        
        print(f"  [Batch: {q_name} vs {len(t_list)} targets] Time: {duration:.5f}s, Perf: {gcups:.4f} GCUPS")
    else:
        print("  [Skipping] Not enough data for batch test.")

if __name__ == "__main__":
    main()
import os
import glob
import time
import sw_cuda_py

# --- Configuration ---
DATA_DIR = "./data"  # Assumes script is in 'benchmarks/' and data is in 'root/data'
MATCH = 2
MISMATCH = -1
GAP = -1

def load_sequences(folder_name):
    """Reads all .seq files in a folder and returns a list of (name, content)."""
    path = os.path.join(DATA_DIR, folder_name, "*.seq")
    files = glob.glob(path)
    sequences = []
    
    if not files:
        print(f"Warning: No .seq files found in {path}")
        return []

    for f in files:
        with open(f, 'r') as file:
            # Read and strip newlines/whitespace
            content = file.read().strip()
            name = os.path.basename(f)
            if content:
                sequences.append((name, content))
    
    print(f"Loaded {len(sequences)} sequences from {folder_name}")
    return sequences

def calculate_gcups(total_cells, elapsed_seconds):
    if elapsed_seconds == 0: return 0.0
    return (total_cells / 1e9) / elapsed_seconds

def run_test(label, func, queries, targets, config, check_ref=None):
    """
    Runs a benchmark for a specific function.
    check_ref: If provided (dict), validates the score against this reference.
    """
    print(f"\n--- Testing: {label} ---")
    
    # 1. Warmup (crucial for GPU to initialize context)
    if "cuda" in label.lower():
        try:
            func(queries[0][1], targets[0][1], config)
        except:
            pass # Just a warmup attempt

    start_time = time.perf_counter()
    
    total_cells = 0
    results = {} # Store results for validation
    
    # Loop over every combination of Query vs Target
    for q_name, q_seq in queries:
        for t_name, t_seq in targets:
            # Call the C++ function
            score = func(q_seq, t_seq, config)
            
            # Metric tracking
            total_cells += len(q_seq) * len(t_seq)
            results[(q_name, t_name)] = score

            # Optional Validation
            if check_ref:
                ref_score = check_ref.get((q_name, t_name))
                if ref_score is not None and score != ref_score:
                    print(f"  [ERROR] Mismatch {q_name} vs {t_name}: Got {score}, Expected {ref_score}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    gcups = calculate_gcups(total_cells, elapsed)
    
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Total Cells: {total_cells}")
    print(f"  Performance: {gcups:.4f} GCUPS")
    
    return results

def main():
    # 1. Setup
    config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    print(f"Configuration: Match={MATCH}, Mismatch={MISMATCH}, Gap={GAP}")
    
    # 2. Load Data
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data. Please ensure 'data/query' and 'data/target' exist and contain .seq files.")
        return

    # 3. Baseline: CPU (We use this to verify correctness of GPU)
    # Note: sw_cpu is slow, so if datasets are huge, you might want to limit this
    cpu_results = run_test("CPU Baseline", sw_cuda_py.sw_cpu, queries, targets, config)
    
    # 4. GPU: Diagonal Wavefront
    run_test("CUDA Diagonal", sw_cuda_py.sw_cuda_diagonal, queries, targets, config, check_ref=cpu_results)
    
    # 5. GPU: Tiled
    run_test("CUDA Tiled", sw_cuda_py.sw_cuda_tiled, queries, targets, config, check_ref=cpu_results)

    # 6. GPU: One-to-Many (Batch) - The fastest method!
    # We must restructure data for the batch interface
    print(f"\n--- Testing: CUDA Batch (O2M) ---")
    q_seq = queries[0][1] # Take first query
    t_list = [t[1] for t in targets] # List of all targets
    
    start = time.perf_counter()
    scores = sw_cuda_py.sw_cuda_o2m(q_seq, t_list, config)
    end = time.perf_counter()
    
    batch_cells = len(q_seq) * sum(len(t) for t in t_list)
    gcups = calculate_gcups(batch_cells, end - start)
    print(f"  Time: {end - start:.4f}s")
    print(f"  Performance: {gcups:.4f} GCUPS")

if __name__ == "__main__":
    main()
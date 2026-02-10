import os
import glob
import time
import sw_cuda_py

# --- Configuration ---
DATA_DIR = "./data"
MATCH = 3
MISMATCH = -3
GAP = -1

def load_sequences(folder_name):
    """
    Reads .seq, .fasta, and .fa files in a folder.
    Returns a list of (name, content).
    """
    folder_path = os.path.join(DATA_DIR, folder_name)
    sequences = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Directory {folder_path} not found")
        return []

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

def run_cpu_baseline_for_batch(queries, targets, config):
    """
    Runs CPU version sequentially to get a baseline for the batch comparison.
    Returns a dict: { query_name: {'time': float, 'cells': int} }
    """
    print("\n--- Pre-calculating CPU Baseline ---")
    results = {}
    
    for q_name, q_seq in queries:
        batch_start = time.perf_counter()
        batch_cells = 0
        
        for t_name, t_seq in targets:
            sw_cuda_py.sw_cpu(q_seq, t_seq, config)
            batch_cells += len(q_seq) * len(t_seq)
            
        batch_end = time.perf_counter()
        
        results[q_name] = {
            'time': batch_end - batch_start,
            'cells': batch_cells
        }
        print(f"  [CPU] Processed batch for {q_name} ({batch_cells} cells)")
        
    return results

def print_batch_table(batch_stats):
    """
    Prints the summary table for One-to-Many execution.
    """
    if not batch_stats:
        print("\nNo data to summarize.")
        return

    # Headers: Total Cells | CPU(s) | CPU GCUPS | O2M(s) | O2M GCUPS
    headers = [
        "Batch Cells", 
        "CPU (s)", "CPU GCUPS", 
        "O2M (s)", "O2M GCUPS"
    ]
    
    col_widths = [15, 12, 12, 12, 12]
    
    header_fmt = " | ".join([f"{{:>{w}}}" for w in col_widths])
    row_fmt    = " | ".join([f"{{:>{w}}}" for w in col_widths])
    sep        = "-+-".join(["-" * w for w in col_widths])
    
    print("\n" + "="*80)
    print("ONE-TO-MANY (BATCH) PERFORMANCE SUMMARY")
    print("="*80)
    print(header_fmt.format(*headers))
    print(sep)

    # Sort by number of cells
    sorted_stats = sorted(batch_stats, key=lambda x: x['cells'])

    for row in sorted_stats:
        cells = row['cells']
        cpu_t = row['cpu_time']
        cpu_g = calculate_gcups(cells, cpu_t)
        o2m_t = row['o2m_time']
        o2m_g = calculate_gcups(cells, o2m_t)

        print(row_fmt.format(
            f"{cells:,}",
            f"{cpu_t:.4f}", f"{cpu_g:.2f}",
            f"{o2m_t:.4f}", f"{o2m_g:.2f}"
        ))
    
    print("="*80 + "\n")

def main():
    config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    print(f"Configuration: Match={MATCH}, Mismatch={MISMATCH}, Gap={GAP}")
    
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data.")
        return

    # 1. Run CPU Baseline (to have something to compare against)
    cpu_stats = run_cpu_baseline_for_batch(queries, targets, config)

    # 2. Run CUDA O2M (One-to-Many)
    print("\n--- Testing: CUDA Batch (O2M) ---")
    
    # Prepare list of target sequences (strings only)
    t_list = [t[1] for t in targets]
    
    # Warmup
    try:
        sw_cuda_py.sw_cuda_o2m(queries[0][1], t_list, config)
    except Exception as e:
        print(f"  [Warmup Failed] {e}")

    batch_results = []

    for q_name, q_seq in queries:
        # Calculate total cells for this batch (Query vs All Targets)
        total_cells = len(q_seq) * sum(len(t) for t in t_list)

        start = time.perf_counter()
        # The Kernel Call
        _ = sw_cuda_py.sw_cuda_o2m(q_seq, t_list, config)
        end = time.perf_counter()
        
        duration = end - start
        gcups = calculate_gcups(total_cells, duration)
        
        print(f"  [Batch: {q_name}] Time: {duration:.5f}s, Perf: {gcups:.4f} GCUPS")
        
        # Store data for the table
        batch_results.append({
            'name': q_name,
            'cells': total_cells,
            'o2m_time': duration,
            'cpu_time': cpu_stats[q_name]['time']  # Retrieve pre-calculated CPU time
        })

    # 3. Print Table
    print_batch_table(batch_results)

if __name__ == "__main__":
    main()
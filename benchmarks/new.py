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

class SWConfig:
    def __init__(self, match, mismatch, gap):
        self.match_score = match
        self.mismatch_score = mismatch
        self.gap_score = gap

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
        # Simple parser for demonstration
        current_name = os.path.basename(f_path)
        with open(f_path, 'r') as file:
            lines = file.readlines()
        
        # Check if FASTA
        if lines and lines[0].startswith('>'):
            seq_acc = ""
            header = current_name
            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    if seq_acc:
                        sequences.append((header, seq_acc))
                        seq_acc = ""
                    header = line[1:]
                else:
                    seq_acc += line
            if seq_acc:
                sequences.append((header, seq_acc))
        else:
            # Assume raw sequence file
            content = "".join([l.strip() for l in lines])
            sequences.append((current_name, content))
    
    print(f"Loaded {len(sequences)} sequences from {folder_name}")
    return sequences

def calculate_gcups(total_cells, elapsed_seconds):
    if elapsed_seconds <= 0: return 0.0
    return (total_cells / 1e9) / elapsed_seconds

# --- Biopython Wrapper ---
def biopython_sw(q, t, config):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = config.match_score
    aligner.mismatch_score = config.mismatch_score
    aligner.open_gap_score = config.gap_score
    aligner.extend_gap_score = config.gap_score
    return int(aligner.score(q, t))

def run_test_and_record(label, method_key, func, queries, targets, config, stats_db, check_ref=None):
    """
    Runs the test and records Time, GCUPS, and Memory (if available) into stats_db.
    """
    print(f"\n--- Testing: {label} ---")
    
    if not queries or not targets:
        print("  [Skipping] Missing queries or targets.")
        return {}

    # Warmup for CUDA (if applicable)
    if "cuda" in label.lower() and queries and targets:
        try:
            func(queries[0][1], targets[0][1], config)
        except Exception as e:
            # Might fail if seqs are too small/large or other issues, just ignore warmup
            pass

    results_scores = {} 
    
    for q_name, q_seq in queries:
        for t_name, t_seq in targets:
            pair_key = (q_name, t_name)
            
            # Initialize storage for this pair if new
            if pair_key not in stats_db:
                stats_db[pair_key] = {
                    'cells': len(q_seq) * len(t_seq),
                    'label': f"{q_name} vs {t_name}",
                    'data': {} # Sub-dictionary for method results
                }

            t_start = time.perf_counter()
            ret_val = func(q_seq, t_seq, config)
            t_end = time.perf_counter()
            
            # Handle return types: int (score only) or tuple (score, peak_bytes)
            score = 0
            mem_kb = 0.0
            
            if isinstance(ret_val, tuple) or isinstance(ret_val, list):
                score = ret_val[0]
                if len(ret_val) > 1:
                    mem_kb = ret_val[1] / (1024.0)
            else:
                score = ret_val
                mem_kb = 0.0 # Not available/applicable

            duration = t_end - t_start
            cells = stats_db[pair_key]['cells']
            gcups = calculate_gcups(cells, duration)
            
            # Store Metrics
            stats_db[pair_key]['data'][method_key] = {
                'time': duration,
                'gcups': gcups,
                'mem_kb': mem_kb,
                'score': score
            }
            
            mem_str = f", Mem: {mem_kb:.2f} KB" if mem_kb > 0 else ""
            print(f"  [{q_name} vs {t_name}] Score: {score}, Time: {duration:.5f}s, Perf: {gcups:.4f} GCUPS{mem_str}")

            results_scores[pair_key] = score

            if check_ref:
                ref_score = check_ref.get(pair_key)
                if ref_score is not None and int(score) != int(ref_score):
                    print(f"    >>> [ERROR] Mismatch! Expected {ref_score}, Got {score}")

    return results_scores

def print_summary_table(stats_db):
    """
    Prints a formatted table including Memory usage for GPU methods.
    """
    if not stats_db:
        print("\nNo data to summarize.")
        return

    # 1. Define Headers
    # We include Mem columns for Diag and Tiled
    headers = [
        "Total Cells", 
        "CPU (s)", "CPU GCUPS",
        "Bio (s)", "Bio GCUPS",
        "Diag (s)", "Diag GCUPS", "Diag KB",
        "Tiled (s)", "Tiled GCUPS", "Tiled KB"
    ]
    
    # 2. Define Column Widths
    col_widths = [15, 10, 10, 10, 10, 10, 10, 10, 10, 12, 10]
    
    # 3. Create format strings
    header_fmt = " | ".join([f"{{:>{w}}}" for w in col_widths])
    row_fmt    = " | ".join([f"{{:>{w}}}" for w in col_widths])
    sep        = "-+-".join(["-" * w for w in col_widths])
    
    print("\n" + "="*145)
    print("PERFORMANCE SUMMARY TABLE (KB = Peak GPU Memory)")
    print("="*145)
    print(header_fmt.format(*headers))
    print(sep)

    # 4. Sort by Total Cells (ascending)
    sorted_items = sorted(stats_db.items(), key=lambda x: x[1]['cells'])

    for pair_key, info in sorted_items:
        total_cells = info['cells']
        methods = info['data']
        
        # Helper to extract data
        def get_vals(key, include_mem=False):
            if key in methods:
                t = methods[key]['time']
                g = methods[key]['gcups']
                val_str = [f"{t:.4f}", f"{g:.2f}"]
                if include_mem:
                    m = methods[key]['mem_kb']
                    val_str.append(f"{m:.1f}")
                return val_str
            else:
                return ["-", "-"] if not include_mem else ["-", "-", "-"]

        c_vals = get_vals("CPU")
        b_vals = get_vals("Bio")
        d_vals = get_vals("Diag", include_mem=True)
        t_vals = get_vals("Tiled", include_mem=True)

        cell_str = f"{total_cells:,}"
        
        # Flatten list for formatting
        row_data = [cell_str] + c_vals + b_vals + d_vals + t_vals

        print(row_fmt.format(*row_data))
    print("="*145 + "\n")

def main():
    # Use the Python wrapper class if available in sw_cuda_py, otherwise local class
    try:
        config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    except AttributeError:
        config = SWConfig(MATCH, MISMATCH, GAP)

    print(f"Configuration: Match={MATCH}, Mismatch={MISMATCH}, Gap={GAP}")
    
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data (ensure ./data/query and ./data/target exist).")
        return

    # Master dictionary
    stats_db = {}

    # 1. Baseline: Custom C++ CPU (Assumes returns int)
    if hasattr(sw_cuda_py, 'sw_cpu'):
        cpu_results = run_test_and_record("CPU Baseline (C++)", "CPU", sw_cuda_py.sw_cpu, queries, targets, config, stats_db)
    else:
        print("Warning: sw_cpu not found in module.")
        cpu_results = {}

    # 2. Baseline: Biopython (Returns int)
    if HAS_BIOPYTHON:
        run_test_and_record("Biopython (CPU)", "Bio", biopython_sw, queries, targets, config, stats_db, check_ref=cpu_results)

    # 3. GPU: Diagonal (Returns tuple: int, size_t)
    if hasattr(sw_cuda_py, 'sw_cuda_diagonal'):
        run_test_and_record("CUDA Diagonal", "Diag", sw_cuda_py.sw_cuda_diagonal, queries, targets, config, stats_db, check_ref=cpu_results)
    
    # 4. GPU: Tiled (Returns tuple: int, size_t)
    if hasattr(sw_cuda_py, 'sw_cuda_tiled'):
        run_test_and_record("CUDA Tiled", "Tiled", sw_cuda_py.sw_cuda_tiled, queries, targets, config, stats_db, check_ref=cpu_results)

    # Print Table
    print_summary_table(stats_db)

if __name__ == "__main__":
    main()
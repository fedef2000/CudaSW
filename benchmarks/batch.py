import os
import glob
import time

# --- Check for Modules ---
try:
    import sw_cuda_py
    HAS_CUDA_MODULE = True
except ImportError:
    HAS_CUDA_MODULE = False
    print("Warning: 'sw_cuda_py' module not found. Only Biopython will be tested if available.")

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
    Returns a list of tuples: (name, content).
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

# --- Benchmark Helper ---
def run_benchmark_for_query(q_name, q_seq, targets, implementations, config, baseline_refs=None):
    """
    Runs all provided implementations for a SINGLE query against ALL targets.
    Returns a dictionary: { "ImplName": (time_s, gcups) }
    """
    results = {}
    target_seqs = [t[1] for t in targets]
    
    # Calculate Total Cells for this Query vs All Targets
    total_cells = sum(len(q_seq) * len(t_seq) for t_seq in target_seqs)

    for impl_name, impl_func in implementations.items():
        start_time = time.perf_counter()
        
        # --- Strategy 1: Batch (One-to-Many) ---
        if "Batch" in impl_name or "O2M" in impl_name:
            # Pass list of targets directly
            # impl_func(query, [list_of_targets], config)
            batch_scores = impl_func(q_seq, target_seqs, config)
            
            # Validation
            if baseline_refs:
                for idx, t_tuple in enumerate(targets):
                    t_name = t_tuple[0]
                    ref = baseline_refs.get(t_name)
                    # batch_scores is a list of ints
                    if ref is not None and int(batch_scores[idx]) != int(ref):
                        print(f"    [ERR] {impl_name} Mismatch {q_name} vs {t_name}: Exp {ref}, Got {batch_scores[idx]}")

        # --- Strategy 2: Sequential (One-to-One) ---
        else:
            # Loop manually
            for t_tuple in targets:
                t_name, t_seq = t_tuple
                res = impl_func(q_seq, t_seq, config)
                
                # Handle tuple return (score, memory)
                score = res[0] if isinstance(res, (tuple, list)) else res

                # Validation
                if baseline_refs:
                    ref = baseline_refs.get(t_name)
                    if ref is not None and int(score) != int(ref):
                        print(f"    [ERR] {impl_name} Mismatch {q_name} vs {t_name}: Exp {ref}, Got {score}")

        end_time = time.perf_counter()
        duration = end_time - start_time
        gcups = calculate_gcups(total_cells, duration)
        
        results[impl_name] = (duration, gcups)

    return total_cells, results

def main():
    # Setup Config
    if HAS_CUDA_MODULE and hasattr(sw_cuda_py, 'SWConfig'):
        config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    else:
        config = SWConfig(MATCH, MISMATCH, GAP)

    print(f"--- SW Benchmark Configuration ---")
    print(f"Match: {MATCH}, Mismatch: {MISMATCH}, Gap: {GAP}")
    
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data.")
        return

    # Define implementations to test
    implementations = {}
    
    if HAS_BIOPYTHON:
        implementations["Bio"] = biopython_sw
    
    if HAS_CUDA_MODULE:
        if hasattr(sw_cuda_py, 'sw_cuda_diagonal'):
            implementations["Diag"] = sw_cuda_py.sw_cuda_diagonal
        if hasattr(sw_cuda_py, 'sw_cuda_tiled'):
            implementations["Tiled"] = sw_cuda_py.sw_cuda_tiled
        if hasattr(sw_cuda_py, 'sw_cuda_o2m'):
            implementations["Batch"] = sw_cuda_py.sw_cuda_o2m

    # Store results per query
    # Structure: [ { "Query": name, "Cells": num, "Results": { "Bio": (t, g), ... } }, ... ]
    all_query_data = []

    print(f"\nStarting Deep Benchmark on {len(queries)} queries vs {len(targets)} targets...\n")

    for q_idx, (q_name, q_seq) in enumerate(queries):
        print(f"[{q_idx+1}/{len(queries)}] Processing Query: {q_name} (len={len(q_seq)})...")
        
        # 1. Generate Baseline for this Query (CPU C++)
        # We need this to validate the other implementations
        baseline_refs = {}
        if HAS_CUDA_MODULE and hasattr(sw_cuda_py, 'sw_cpu'):
             for t_name, t_seq in targets:
                 baseline_refs[t_name] = sw_cuda_py.sw_cpu(q_seq, t_seq, config)
        
        # 2. Run Benchmarks
        total_cells, res_dict = run_benchmark_for_query(
            q_name, q_seq, targets, implementations, config, baseline_refs
        )
        
        # 3. Add CPU Baseline to results manually for the table
        if HAS_CUDA_MODULE and hasattr(sw_cuda_py, 'sw_cpu'):
             # Measure CPU time again purely for stats
             t0 = time.perf_counter()
             for t_name, t_seq in targets:
                 sw_cuda_py.sw_cpu(q_seq, t_seq, config)
             t1 = time.perf_counter()
             res_dict["CPU"] = (t1-t0, calculate_gcups(total_cells, t1-t0))

        all_query_data.append({
            "Query": q_name,
            "Cells": total_cells,
            "Results": res_dict
        })

    # --- Print Summary Table ---
    # We want columns: Query, Total Cells, then GCUPS for each method
    
    # 1. Get list of all method names found across results
    method_names = []
    # Collect unique keys, sorting CPU first if present, then others
    if all_query_data:
        keys = list(all_query_data[0]["Results"].keys())
        if "CPU" in keys: keys.remove("CPU"); method_names.append("CPU")
        if "Bio" in keys: keys.remove("Bio"); method_names.append("Bio")
        method_names.extend(sorted(keys)) # Sort remaining (Diag, Tiled, Batch)

    # 2. Header
    # Name width: 20, Cells width: 12, Method width: 10
    header_fmt = "{:<20} | {:>12} " + "".join([f"| {{:^{len(m)+2}}} " for m in method_names])
    row_fmt    = "{:<20} | {:>12,} " + "".join([f"| {{:>{len(m)+2}.2f}} " for m in method_names])

    print("\n" + "="* (35 + 13 * len(method_names)))
    print("PERFORMANCE SUMMARY (GCUPS)")
    print("="* (35 + 13 * len(method_names)))
    
    # Print Header
    print(header_fmt.format("Query Name", "Total Cells", *method_names))
    print("-" * (35 + 13 * len(method_names)))

    # Print Rows
    for entry in all_query_data:
        vals = []
        for m in method_names:
            if m in entry["Results"]:
                vals.append(entry["Results"][m][1]) # Append GCUPS
            else:
                vals.append(0.0)
        
        print(row_fmt.format(entry["Query"], entry["Cells"], *vals))

    print("="* (35 + 13 * len(method_names)) + "\n")

if __name__ == "__main__":
    main()
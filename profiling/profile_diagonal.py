import os
import glob
import time
import sw_cuda_py


# --- Configuration ---
DATA_DIR = "../benchmarks/data"
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

def run_test(label, func, queries, targets, config, check_ref=None):
    print(f"\n--- Testing: {label} ---")
    
    if not queries or not targets:
        print("  [Skipping] Missing queries or targets.")
        return {}
    
    for q_name, q_seq in queries:
        for t_name, t_seq in targets:                
            score = func(q_seq, t_seq, config)

        
    print(f"  --- {label} DONE ---")
    
    return {}

def main():
    config = sw_cuda_py.SWConfig(MATCH, MISMATCH, GAP)
    print(f"Configuration: Match={MATCH}, Mismatch={MISMATCH}, Gap={GAP}")
    
    queries = load_sequences("query")
    targets = load_sequences("target")
    
    if not queries or not targets:
        print("Error: Missing data.")
        return

    # GPU: Diagonal
    run_test("CUDA Diagonal", sw_cuda_py.sw_cuda_diagonal, queries, targets, config)
    
if __name__ == "__main__":
    main()
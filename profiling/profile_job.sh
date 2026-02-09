#!/bin/bash
#SBATCH --job-name=GPU_align_profile
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Modificato: Il tuo codice è single-process
#SBATCH --gres=gpu:1                 # Modificato: Ti serve solo 1 GPU
#SBATCH --cpus-per-task=4            # 4 CPU per gestire I/O e driver sono sufficienti
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=profile.log

# --- CONFIGURAZIONE ---
PROFILE_TOOL="nsys" # "nsys" per timeline globale, "ncu" per analisi kernel profonda
APP_SRC="opt_gpu_version.cu"
APP_EXE="gpu_profile"

# --- CARICAMENTO MODULI ---
module purge
module load gcc/12.4.0
module load nvhpc/25.1

# Setup CUDA HOME (come nel tuo script originale)
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${NVHPC:-}" ]; then
    CUDA_HOME="$NVHPC/Linux_x86_64/25.1/cuda"
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname $(dirname $(command -v nvcc)))"
  fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- 1. GENERAZIONE DATI DI TEST (Se non esistono) ---
# Il tuo codice richiede a.seq e b.seq. Li creiamo random se mancano
# per assicurare che il profiler abbia qualcosa da macinare.
if [ ! -f "a.seq" ] || [ ! -f "b.seq" ]; then
    echo "Generazione file sequenze di test..."
    # Genera due sequenze casuali di 15.000 caratteri (A,C,G,T)
    tr -dc 'ACGT' </dev/urandom | head -c 15000 > a.seq
    echo "" >> a.seq # newline
    tr -dc 'ACGT' </dev/urandom | head -c 15000 > b.seq
    echo "" >> b.seq # newline
fi

# --- 2. COMPILAZIONE ---
echo "Compilazione di $APP_SRC..."
# -O3 per ottimizzazione host, -arch=sm_70 per V100
nvcc -O3 -arch=sm_80 "$APP_SRC" -o "$APP_EXE"

if [ ! -f "$APP_EXE" ]; then
    echo "Errore: Compilazione fallita."
    exit 1
fi

# --- 3. PROFILAZIONE ---
echo "Avvio profilazione con $PROFILE_TOOL..."

if [ "$PROFILE_TOOL" = "nsys" ]; then
    # Nsight Systems: Ottimo per vedere trasferimenti memoria vs calcolo kernel
    # Rimosso 'mpi' dal trace perché non lo usi
    nsys profile --force-overwrite=true \
    -o "${APP_EXE}_nsys" \
    --trace="cuda,osrt,nvtx" \
    --stats=true \
    ./"$APP_EXE"
    
    # Genera summary leggibile
    # (Nota: nelle versioni recenti --stats=true sopra lo fa già, ma questo crea un file separato)
    nsys stats "${APP_EXE}_nsys.nsys-rep" > "${APP_EXE}_nsys_summary.txt"

elif [ "$PROFILE_TOOL" = "ncu" ]; then
    # Nsight Compute: Ottimo per analizzare il kernel "compute_diagonal"
    # Set "full" è molto pesante, usiamo sezioni specifiche per iniziare
    ncu --set full \
    --target-processes all \
    --export "${APP_EXE}_ncu" \
    --force-overwrite \
    ./"$APP_EXE"
fi

echo "Job completato."
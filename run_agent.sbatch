#!/bin/bash
#SBATCH --job-name=drug-agent
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=agent-%j.out
#SBATCH --error=agent-%j.err

set -euo pipefail

# Load Conda
eval "$(conda shell.bash hook)"
conda activate drugdesign

# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Separate docking environment
export DOCKING_PYTHON="${DOCKING_PYTHON:-$HOME/.conda/envs/docking/bin/python}"

# Verify required environment variables
: "${NRP_API_KEY:?Export NRP_API_KEY before submitting}"
: "${JAK2_MODEL_PATH:?Export JAK2_MODEL_PATH before submitting}"
: "${LCK_MODEL_PATH:?Export LCK_MODEL_PATH before submitting}"

echo "=================================="
echo "JAK2 DRUG DESIGN EXPERIMENT"
echo "=================================="
echo "Running on: $(hostname)"
echo "Python: $(command -v python)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: Disabled"
echo "Start time: $(date)"

# Verify Chemprop/PyTorch environment
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'

# Verify docking environment
"$DOCKING_PYTHON" -c \
'import sys, dockstring; print("Docking Python:", sys.version.split()[0], "DOCKSTRING:", dockstring.__version__)'

# Run molecular design agent
python agent.py \
    --iterations 10 \
    --candidates 20 \
    --dock-top 5 \
    --run-docking

echo "=================================="
echo "EXPERIMENT COMPLETED"
echo "End time: $(date)"
echo "=================================="

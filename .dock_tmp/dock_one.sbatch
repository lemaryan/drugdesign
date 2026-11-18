#!/bin/bash
#SBATCH --job-name=dock-one
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate drugdesign
export CUDA_VISIBLE_DEVICES=""
export DOCKING_PYTHON="${DOCKING_PYTHON:-$HOME/.conda/envs/docking/bin/python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd /global/home/users/maryanle/drug_design_clean
echo "obabel: $(obabel -V 2>&1 | head -1)"
python -s "$SCRATCH_DIR/dock_one.py" "$SMILES"

#!/bin/bash
#SBATCH --job-name=redock-seeds
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=redock-%j.out
#SBATCH --error=redock-%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"

# Activating drugdesign puts Open Babel 3.1.0 on PATH, which is what the
# original run used. The docking env's obabel was upgraded to 3.2.1 and
# DOCKSTRING rejects it, so do not activate the docking env here.
conda activate drugdesign

export CUDA_VISIBLE_DEVICES=""
export DOCKING_PYTHON="${DOCKING_PYTHON:-$HOME/.conda/envs/docking/bin/python}"

# Vina gets its parallelism from --num_cpus, so keep the math libraries serial.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "obabel: $(obabel -V 2>&1 | head -1)"
echo "Start: $(date)"

"$DOCKING_PYTHON" -s redock_seeds.py \
    --targets data/redock_targets.csv \
    --out data/redock_seeds.csv \
    --seeds 974528263,20260101,20260102,20260103,20260104 \
    --workers 4

echo "End: $(date)"

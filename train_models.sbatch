#!/bin/bash
#SBATCH --job-name=chemprop
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=chemprop-%j.out

set -euo pipefail

# Run `sbatch train_models.sbatch` from the project directory on Savio.
# Initialize Conda for this non-interactive batch shell, then activate the
# existing Chemprop environment.
eval "$(conda shell.bash hook)"
conda activate drugdesign

echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname)"
echo "Python: $(command -v python)"
python -c 'import chemprop; print("Chemprop:", chemprop.__version__)'

python train_models.py \
  --max-rows 50000 \
  --message-hidden-dim 4096 \
  --ffn-hidden-dim 4096 \
  --depth 9 \
  --ffn-num-layers 4 \
  --epochs 10 \
  --output-dir "$PWD/models/trained/moderate_${SLURM_JOB_ID:-local}"

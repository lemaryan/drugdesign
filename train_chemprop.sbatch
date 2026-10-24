#!/bin/bash
#SBATCH --job-name=chemprop
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_ica
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=chemprop-%j.out
#SBATCH --chdir=/global/home/users/maryanle/drug_design_clean

set -euo pipefail
source ~/.bashrc
conda activate drugdesign

python train_models.py \
  --message-hidden-dim 1000 \
  --ffn-hidden-dim 1000 \
  --depth 5 \
  --ffn-num-layers 2 \
  --epochs 1 \
  --output-dir "$PWD/models/trained/batch_run"


#!/usr/bin/env bash
#SBATCH --job-name=ntfm-pd
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=25-00:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
workflows/slurm/launch_training.sh configs/training/clinical_pd.yaml

#!/usr/bin/env bash
set -euo pipefail
config="${1:?training config required}"
: "${SLURM_NNODES:=4}"
: "${SLURM_GPUS_ON_NODE:=8}"
master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
master_port="${MASTER_PORT:-29400}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_ASYNC_ERROR_HANDLING=1 NCCL_NVLS_ENABLE=1 TORCH_NCCL_BLOCKING_WAIT=0
export MASTER_ADDR="$master_addr" MASTER_PORT="$master_port"
srun --nodes="$SLURM_NNODES" --ntasks="$((SLURM_NNODES * SLURM_GPUS_ON_NODE))" \
  --ntasks-per-node="$SLURM_GPUS_ON_NODE" --gpus-per-task=1 \
  build/bin/ntfm-train --config "$config"

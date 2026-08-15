#!/usr/bin/env bash
set -euo pipefail
request="${1:?request JSON required}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" mpirun -np 2 build/bin/ntfm-infer --config configs/deployment/product_b200x2.yaml --request "$request"

#!/usr/bin/env bash
set -euo pipefail
manifest="${1:?manifest JSONL required}"
gpus="${2:-0,1,2,3,4,5,6,7}"
build/bin/ntfm-tool prepare-packs --manifest "$manifest" --compiler build/bin/neurocompile --config configs/compiler/neurocompiler.yaml --gpus "$gpus" --log compile-results.jsonl

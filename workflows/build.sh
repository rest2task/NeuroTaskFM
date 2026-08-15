#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=100a -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/libtorch}"
cmake --build build -j "${NPROC:-32}"
mkdir -p bin
go build -trimpath -ldflags '-s -w' -o bin/ntfm ./src/platform/cmd/ntfm
go build -trimpath -ldflags '-s -w' -o bin/neurotaskd ./src/platform/cmd/neurotaskd
go build -trimpath -ldflags '-s -w' -o bin/neurocompiler-agent ./src/platform/cmd/neurocompiler-agent

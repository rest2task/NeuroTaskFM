#!/usr/bin/env bash
set -euo pipefail
diagnostics="${1:?probe diagnostics JSON required}"
round="${2:-round-001}"
mkdir -p "artifacts/search/$round"
bin/neurocompiler-agent --config configs/agent/gemini.json --diagnostics "$diagnostics" --output "artifacts/search/$round/candidate.json"
build/bin/ntfm-tool candidate-to-yaml --candidate "artifacts/search/$round/candidate.json" --template configs/compiler/neurocompiler.yaml --output "artifacts/search/$round/compiler.yaml"
echo "artifacts/search/$round/compiler.yaml"

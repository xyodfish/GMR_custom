#!/usr/bin/env bash
# Quick demo: Online Batch-Lite on one GVHMR clip (headless timing + optional viewer).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PT="${1:-data/gvhmr_test_videos/walking/hmr4d_results.pt}"
PRESET="${PRESET:-balanced}"
ROBOT="${ROBOT:-unitree_g1}"
VIEW="${VIEW:-0}"

if [[ ! -f "$PT" ]]; then
  echo "Missing: $PT" >&2
  exit 1
fi

echo "=== Online Batch-Lite demo ==="
echo "  clip: $PT"
echo "  preset: $PRESET"

python3 scripts/gvhmr/to_robot_online_batch.py \
  --gvhmr_pred_file "$PT" \
  --robot "$ROBOT" \
  --preset "$PRESET" \
  --compare_ik \
  --headless

echo ""
echo "Full benchmark:"
echo "  bash scripts/tools/run_gvhmr_retarget_benchmark.sh"
echo "Report: docs/online_batch_retargeting.md"

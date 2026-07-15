#!/usr/bin/env bash
# Verify C++ batch TO parity: Py quality match + banded ≡ dense solver.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PT_FILE="${PT_FILE:-output/gvhmr_pt/cxk-ball_hmr4d_results.pt}"
ROBOT="${ROBOT:-unitree_g1}"
MAX_FRAMES="${MAX_FRAMES:-120}"
OUTPUT_JSON="${OUTPUT_JSON:-output/batch_opt_quality.json}"
PY_CPP_RMSE_MAX="${PY_CPP_RMSE_MAX:-1e-5}"
BANDED_DENSE_RMSE_MAX="${BANDED_DENSE_RMSE_MAX:-1e-6}"
BUILD="${BUILD:-1}"

if [[ ! -f "$PT_FILE" ]]; then
  echo "ERROR: missing test motion: $PT_FILE" >&2
  echo "Set PT_FILE to a GVHMR .pt file." >&2
  exit 1
fi

if [[ "$BUILD" == "1" ]]; then
  echo "[verify_batch_to_parity] building gmr_batch_to_cli..."
  cmake --build cpp/build -j --target gmr_batch_to_cli
fi

if [[ ! -x cpp/build/gmr_batch_to_cli ]]; then
  echo "ERROR: cpp/build/gmr_batch_to_cli not found or not executable" >&2
  exit 1
fi

if [[ -d /opt/robot/devel/lib ]]; then
  export LD_LIBRARY_PATH="/opt/robot/devel/lib:${LD_LIBRARY_PATH:-}"
fi

echo "[verify_batch_to_parity] running compare_batch_opt_quality.py"
echo "  pt=$PT_FILE robot=$ROBOT max_frames=$MAX_FRAMES"

python scripts/analysis/compare_batch_opt_quality.py \
  --pt_file "$PT_FILE" \
  --robot "$ROBOT" \
  --max_frames "$MAX_FRAMES" \
  --contact_ground \
  --output_json "$OUTPUT_JSON" \
  --py_cpp_rmse_max "$PY_CPP_RMSE_MAX" \
  --banded_dense_rmse_max "$BANDED_DENSE_RMSE_MAX" \
  --fail_on_threshold

echo "[verify_batch_to_parity] OK"

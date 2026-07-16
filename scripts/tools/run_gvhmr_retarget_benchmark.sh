#!/usr/bin/env bash
# Download curated GVHMR test videos + run GVHMR inference + benchmark online/offline retarget methods.
#
# Clip criteria (see data/gvhmr_test_videos/README.md):
#   - single person only
#   - full body visible (head to feet, legs not cropped)
#   - ground-based motion (walking, tennis, etc.; pure walking not required)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VIDEO_DIR="${VIDEO_DIR:-$REPO_ROOT/data/gvhmr_test_videos}"
CLIPS_FILE="${CLIPS_FILE:-$VIDEO_DIR/clips.txt}"
MAX_FRAMES="${MAX_FRAMES:-200}"
ROBOT="${ROBOT:-unitree_g1}"
BUILD="${BUILD:-1}"

mkdir -p "$VIDEO_DIR" output

download_if_missing() {
  local url="$1" dest="$2"
  if [[ ! -f "$dest" ]]; then
    echo "  download $dest"
    curl -fsSL -L -o "$dest" "$url"
  fi
}

echo "[1/4] Curate test videos in $VIDEO_DIR"

# GVHMR official examples (single person, full body)
GVHMR_ROOT="${GVHMR_ROOT:-$HOME/Workspace/xeeform_motion_generation/GVHMR}"
if [[ -d "$GVHMR_ROOT/docs/example_video" ]]; then
  cp -n "$GVHMR_ROOT/docs/example_video/walking.mp4" "$VIDEO_DIR/" 2>/dev/null || true
  cp -n "$GVHMR_ROOT/docs/example_video/tennis.mp4" "$VIDEO_DIR/" 2>/dev/null || true
fi

# Full-body single-person ground-motion samples (static cam friendly)
download_if_missing \
  "https://raw.githubusercontent.com/davidpagnon/Sports2D/main/Sports2D/Demo/demo.mp4" \
  "$VIDEO_DIR/sports2d_walk.mp4"
download_if_missing \
  "https://raw.githubusercontent.com/nihalanas/gait-analysis/main/test_videos/gait_video12.mp4" \
  "$VIDEO_DIR/gait_track_12.mp4"
download_if_missing \
  "https://raw.githubusercontent.com/nihalanas/gait-analysis/main/test_videos/gait_video4.mp4" \
  "$VIDEO_DIR/gait_track_4.mp4"
download_if_missing \
  "https://raw.githubusercontent.com/nihalanas/gait-analysis/main/test_videos/gait_video14.mp4" \
  "$VIDEO_DIR/gait_track_14.mp4"

# Remove clips that do not meet benchmark criteria
for bad in cxk-ball people_walk walk_intel gait_side_1080p tendra_mujoco_test; do
  rm -f "$VIDEO_DIR/${bad}.mp4"
  rm -rf "$VIDEO_DIR/${bad}"
done

if [[ ! -f "$CLIPS_FILE" ]]; then
  echo "Missing allowlist: $CLIPS_FILE" >&2
  exit 1
fi

mapfile -t CLIP_NAMES < <(grep -v '^[[:space:]]*#' "$CLIPS_FILE" | grep -v '^[[:space:]]*$' || true)
if [[ ${#CLIP_NAMES[@]} -eq 0 ]]; then
  echo "No clips listed in $CLIPS_FILE" >&2
  exit 1
fi

VIDEO_ARGS=()
for name in "${CLIP_NAMES[@]}"; do
  path="$VIDEO_DIR/$name"
  if [[ ! -f "$path" ]]; then
    echo "Missing curated clip: $path" >&2
    exit 1
  fi
  VIDEO_ARGS+=("$path")
done

echo "Allowlisted clips:"
ls -lh "${VIDEO_ARGS[@]}"

echo "[2/4] GVHMR inference"
python3 scripts/gvhmr/batch_video_to_gvhmr.py \
  --videos "${VIDEO_ARGS[@]}" \
  --copy_into_video_dir \
  --manifest output/gvhmr_test_manifest.txt

echo "[3/4] Build C++ batch TO CLI (optional)"
if [[ "$BUILD" == "1" ]]; then
  cmake --build cpp/build -j --target gmr_batch_to_cli
fi

if [[ -d /opt/robot/devel/lib ]]; then
  export LD_LIBRARY_PATH="/opt/robot/devel/lib:${LD_LIBRARY_PATH:-}"
fi

PT_ARGS=()
for name in "${CLIP_NAMES[@]}"; do
  stem="${name%.mp4}"
  PT_ARGS+=(--pt_file "data/gvhmr_test_videos/${stem}/hmr4d_results.pt")
done

echo "[4/4] Benchmark online + offline retarget methods"
python3 scripts/analysis/benchmark_gvhmr_retarget_methods.py \
  "${PT_ARGS[@]}" \
  --robot "$ROBOT" \
  --max_frames "$MAX_FRAMES" \
  --contact_ground \
  --methods "ik,online_batch,py_batch_to,cpp_batch_to" \
  --output_json output/gvhmr_retarget_benchmark.json

echo "Done. Report: output/gvhmr_retarget_benchmark.json"

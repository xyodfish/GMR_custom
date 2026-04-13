#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bvh_to_retargeting_frame.sh <bvh_file> [--format <lafan1|nokov>] [--motion_fps <int>] [--out_dir <dir>]

Behavior:
  Output json name is always:
    <bvh_basename>_retarget_frame.json

Example:
  ./scripts/bvh_to_retargeting_frame.sh ~/Downloads/lafan1/fight1_subject5.bvh
  ./scripts/bvh_to_retargeting_frame.sh ./demo.bvh --format nokov --motion_fps 60 --out_dir ./tmp
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

bvh_file="$1"
shift

format="lafan1"
motion_fps="30"
out_dir="$(dirname "$bvh_file")"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      format="$2"
      shift 2
      ;;
    --motion_fps)
      motion_fps="$2"
      shift 2
      ;;
    --out_dir)
      out_dir="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$bvh_file" ]]; then
  echo "BVH file not found: $bvh_file" >&2
  exit 1
fi

mkdir -p "$out_dir"

bvh_name="$(basename "$bvh_file")"
bvh_stem="${bvh_name%.*}"
out_json="${out_dir}/${bvh_stem}_retarget_frame.json"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"

python3 "${script_dir}/bvh_to_retargeting_frame.py" \
  --bvh_file "$bvh_file" \
  --format "$format" \
  --motion_fps "$motion_fps" \
  --save_path "$out_json"

echo "Output: $out_json"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <input.bvh> [format: lafan1|nokov] [motion_fps]" >&2
  exit 1
fi

input_bvh="$1"
format="${2:-lafan1}"
motion_fps="${3:-30}"

if [[ ! -f "$input_bvh" ]]; then
  echo "Error: BVH file not found: $input_bvh" >&2
  exit 1
fi

file_name="$(basename -- "$input_bvh")"
base_name="${file_name%.*}"
output_dir="$(dirname -- "$input_bvh")"
output_json="${output_dir}/${base_name}_retarget_frame.json"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

PYTHONPATH="${repo_root}:${PYTHONPATH:-}" python3 "${repo_root}/scripts/bvh_to_retargeting_frame.py" \
  --bvh_file "$input_bvh" \
  --format "$format" \
  --motion_fps "$motion_fps" \
  --save_path "$output_json"

echo "Output JSON: $output_json"

#!/usr/bin/env python3
"""Offline ground-align stats / dry-run for GVHMR .pt (or SMPL-X / BVH).

Prints float%>5cm before→after. Does not rewrite the .pt by default.

Examples
--------
# Single clip:
python scripts/tools/ground_align_pt.py \\
  --input_file data/gvhmr_test_videos/ma_girl_run/hmr4d_results.pt

# Scan a folder of GVHMR outputs:
python scripts/tools/ground_align_pt.py \\
  --input_dir data/gvhmr_test_videos --mode lower_envelope
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting.ground_align_frames import (  # noqa: E402
    compute_ground_align_offsets,
)
from general_motion_retargeting.human_frame_loaders import (  # noqa: E402
    load_human_motion_frames,
)


def _find_pts(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob("hmr4d_results.pt"))


def _label(path: pathlib.Path) -> str:
    if path.name == "hmr4d_results.pt":
        return path.parent.name
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Recursively find hmr4d_results.pt under this dir",
    )
    parser.add_argument(
        "--mode",
        choices=["lower_envelope", "support_hold"],
        default="lower_envelope",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument(
        "--body_model_dir",
        type=str,
        default=str(REPO / "assets" / "body_models"),
    )
    args = parser.parse_args()

    paths: list[pathlib.Path] = []
    if args.input_file:
        paths.append(pathlib.Path(args.input_file).expanduser().resolve())
    if args.input_dir:
        paths.extend(_find_pts(pathlib.Path(args.input_dir).expanduser().resolve()))
    if not paths:
        parser.error("Provide --input_file and/or --input_dir")

    print(f"{'clip':<28} {'frames':>6} {'float%':>14} {'z_min_mean':>18} {'|off|_max':>10}")
    print("-" * 82)
    for path in paths:
        frames, fps, _h, _src = load_human_motion_frames(
            path,
            body_model_dir=args.body_model_dir,
            tgt_fps=args.motion_fps,
            max_frames=args.max_frames,
            ground_align=False,
        )
        _off, stats = compute_ground_align_offsets(frames, fps, mode=args.mode)
        print(
            f"{_label(path):<28} {len(frames):>6} "
            f"{stats['float_pct_before']:5.0f}→{stats['float_pct_after']:5.0f}% "
            f"{stats['z_min_before_mean']:7.3f}→{stats['z_min_after_mean']:6.3f} "
            f"{stats['offset_max']:10.3f}"
        )


if __name__ == "__main__":
    main()

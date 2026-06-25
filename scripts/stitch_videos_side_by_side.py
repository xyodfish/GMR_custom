"""Stitch two videos into one side-by-side comparison (left | right).

Example:
    python scripts/stitch_videos_side_by_side.py \\
        --left  videos/contact_compare/foo_contact_off.mp4 \\
        --right videos/contact_compare/foo_contact_on.mp4 \\
        --output videos/contact_compare/foo_side_by_side.mp4

Or infer paths from a BVH stem:
    python scripts/stitch_videos_side_by_side.py \\
        --stem multipleActions1_subject3 \\
        --dir videos/contact_compare
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def stitch_videos_side_by_side(
    left_path: str | pathlib.Path,
    right_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    crf: int = 18,
) -> pathlib.Path:
    left_path = pathlib.Path(left_path)
    right_path = pathlib.Path(right_path)
    output_path = pathlib.Path(output_path)

    if not left_path.is_file():
        raise FileNotFoundError(f"Left video not found: {left_path}")
    if not right_path.is_file():
        raise FileNotFoundError(f"Right video not found: {right_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale both sides to the same height, then hstack.
    filter_complex = (
        "[0:v]scale=iw:min(ih\\,ih)[left];"
        "[1:v]scale=iw:min(ih\\,ih)[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left_path),
        "-i",
        str(right_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n"
            + (result.stderr or result.stdout or "unknown error")
        )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a left-right comparison video.")
    parser.add_argument("--left", type=str, default=None, help="Left video path (OFF).")
    parser.add_argument("--right", type=str, default=None, help="Right video path (ON).")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path.")
    parser.add_argument(
        "--dir",
        type=str,
        default="videos/contact_compare",
        help="Directory used with --stem to infer default paths.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="BVH filename stem, e.g. multipleActions1_subject3.",
    )
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality (lower=better).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = pathlib.Path(args.dir)

    if args.stem is not None:
        left = base_dir / f"{args.stem}_contact_off.mp4"
        right = base_dir / f"{args.stem}_contact_on.mp4"
        output = (
            pathlib.Path(args.output)
            if args.output
            else base_dir / f"{args.stem}_side_by_side.mp4"
        )
    else:
        if not args.left or not args.right:
            print("Provide --left and --right, or use --stem.", file=sys.stderr)
            sys.exit(1)
        left = pathlib.Path(args.left)
        right = pathlib.Path(args.right)
        if args.output:
            output = pathlib.Path(args.output)
        else:
            output = left.with_name(f"{left.stem}_side_by_side.mp4")

    out = stitch_videos_side_by_side(left, right, output, crf=args.crf)
    print(f"Saved side-by-side video: {out}")


if __name__ == "__main__":
    main()

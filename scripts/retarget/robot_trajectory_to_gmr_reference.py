#!/usr/bin/env python3
"""Retarget a Unitree G1 joint trajectory to one or more GMR target robots.

Pipeline:
  G1 CSV/qpos JSON/NPY -> MuJoCo FK semantic sites -> canonical SMPL-X proxy
  -> C++ Batch TO -> contact-aware kinematic postprocess -> robot qpos JSON.

Examples:
  python scripts/retarget/robot_trajectory_to_gmr_reference.py \
    --input /path/to/walk.csv --robot-b unitree_h2

  python scripts/retarget/robot_trajectory_to_gmr_reference.py --list-robot-b
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SRC_PYTHON = REPO / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from robot_to_gmr import (  # noqa: E402
    CanonicalTrajectoryFitter,
    SemanticSiteMap,
    SourceTrajectoryReader,
    align_wrists_to_forearm,
    flatten_stance_feet_ik,
    ground_align_frames,
    list_smplx_target_robots,
    measure_stance_foot_slip_mps,
    model_has_wrist_pitch_yaw,
    parse_robot_b_list,
    plant_stance_feet_ik,
    snap_robot_qpos_to_ground,
)


DEFAULT_MAPPING = REPO / "config" / "retarget" / "source" / "unitree_g1_to_smplx_proxy.yaml"
DEFAULT_BATCH_CLI = REPO / "cpp" / "build" / "gmr_batch_to_cli"
DEFAULT_OUT_DIR = REPO / "output" / "robot_to_gmr"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _clip_name(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".qpos.json"):
        return name[: -len(".qpos.json")]

    return input_path.stem


def _reference_payload(
    trajectory,
    frames: list[dict],
    contacts: list[dict],
    quality,
    site_map,
    *,
    clip: str,
    mapping: Path,
    ground_aligned: bool,
) -> dict:
    return {
        "schema_version": "gmr_reference_v1",
        "fps": float(trajectory.fps),
        "src_human": "smplx",
        "actual_human_height": float(site_map.canonical_height),
        "source": {
            "robot": "unitree_g1",
            "model_hash": trajectory.model_hash,
            "trajectory_id": clip,
            "root_type": trajectory.root_type,
            "input_path": trajectory.metadata["input_path"],
            "global_scale": float(site_map.global_scale),
            "ground_aligned": ground_aligned,
        },
        "canonical": {
            "model": "smplx_neutral_1p8m",
            "mapping_version": mapping.name,
            "coordinate_system": "right_handed_z_up_m_wxyz",
            "height_m": float(site_map.canonical_height),
        },
        "contacts": contacts,
        "frames": frames,
        "quality": asdict(quality),
    }


def _source_qpos_payload(trajectory) -> dict:
    return {
        "robot": "unitree_g1",
        "fps": float(trajectory.fps),
        "nq": int(trajectory.qpos_frames.shape[1]),
        "num_frames": trajectory.num_frames,
        "source_id": trajectory.source_id,
        "qpos_frames": trajectory.qpos_frames.tolist(),
    }


def _run_batch_to(
    *,
    cpp_cli: Path,
    human_json: Path,
    out_json: Path,
    robot: str,
    fast: bool,
    verbose: bool,
    parallel: bool,
    contact_ground: bool,
    joint_limit_margin_deg: float,
) -> None:
    if not cpp_cli.is_file():
        raise FileNotFoundError(
            f"C++ Batch TO executable not found: {cpp_cli}. "
            "Build it with: cmake --build cpp/build -j --target gmr_batch_to_cli"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(cpp_cli),
        "--gmr_root",
        str(REPO),
        "--robot",
        robot,
        "--human_frame_json",
        str(human_json),
        "--out_json",
        str(out_json),
        "--src_human",
        "smplx",
        "--window_size",
        "16",
        "--window_stride",
        "8",
        "--joint_limit_margin_deg",
        str(joint_limit_margin_deg),
    ]
    if fast:
        command.append("--fast")

    if verbose:
        command.append("--verbose")

    if parallel:
        command.append("--parallel")

    if contact_ground:
        command.append("--contact_ground")

    env = os.environ.copy()
    library_dirs = [Path("/opt/robot/devel/x86_64_gcc114/lib"), Path("/opt/robot/devel/lib")]
    available = [str(path) for path in library_dirs if path.is_dir()]
    if available:
        available.append(env.get("LD_LIBRARY_PATH", ""))
        env["LD_LIBRARY_PATH"] = ":".join(available)

    print("[robot-to-gmr]", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO, env=env)


def _postprocess(
    qpos_frames: np.ndarray,
    *,
    model_xml: Path,
    contacts: list[dict[str, bool]],
    fps: float,
    planar_base: bool,
    align_wrists: bool,
    mode: str,
) -> tuple[np.ndarray, dict]:
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    meta: dict = {"mode": mode, "planar_base": planar_base}
    if mode == "none":
        meta["skipped"] = True
        return qpos, meta

    if planar_base:
        meta["skipped_free_root_postprocess"] = True
        return qpos, meta

    model_xml_s = str(model_xml)
    slip_before = measure_stance_foot_slip_mps(qpos, model_xml_s, contacts, fps)
    qpos = plant_stance_feet_ik(qpos, model_xml_s, contacts)
    qpos = flatten_stance_feet_ik(qpos, model_xml_s, contacts)
    qpos = snap_robot_qpos_to_ground(qpos, model_xml_s, contacts=contacts)
    if align_wrists:
        qpos = align_wrists_to_forearm(qpos, model_xml_s)

    slip_after = measure_stance_foot_slip_mps(qpos, model_xml_s, contacts, fps)
    meta.update(
        {
            "stance_feet_planted_ik": True,
            "stance_feet_flattened_ik": True,
            "ground_snapped_on_contact": True,
            "airborne_root_height_preserved": True,
            "airborne_sole_penetration_prevented": True,
            "wrists_aligned_to_forearm": align_wrists,
            "stance_slip_mps": {"before": slip_before, "after": slip_after},
        }
    )
    return qpos, meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Unitree G1 CSV, qpos JSON, or NPY")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--robot-b",
        default="unitree_h2",
        help="Target robot, comma-separated targets, or 'all'",
    )
    parser.add_argument("--list-robot-b", action="store_true")
    parser.add_argument("--skip-robot-b", action="store_true")
    parser.add_argument("--batch-cli", type=Path, default=DEFAULT_BATCH_CLI)
    parser.add_argument("--postprocess", choices=("none", "minimal"), default="minimal")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--no-ground-align", action="store_true")
    parser.add_argument("--no-contact-ground", action="store_true")
    parser.add_argument("--no-align-wrists", action="store_true")
    parser.add_argument("--joint-limit-margin-deg", type=float, default=0.0)
    args = parser.parse_args()

    if not args.list_robot_b and args.input is None:
        parser.error("--input is required unless --list-robot-b is used")

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    return args


def main() -> int:
    args = _parse_args()
    if args.list_robot_b:
        for robot in list_smplx_target_robots(REPO):
            print(robot)

        return 0

    input_path = args.input.expanduser().resolve()
    mapping = args.mapping.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    clip = _clip_name(input_path)

    reader = SourceTrajectoryReader(mapping, REPO)
    trajectory = reader.load(input_path, fps=args.fps, max_frames=args.max_frames)
    site_map = SemanticSiteMap(mapping, REPO)
    semantic_frames = site_map.extract_sequence(trajectory)
    canonical_fitter = CanonicalTrajectoryFitter(mapping)
    frames, contacts, quality = canonical_fitter.fit(semantic_frames, trajectory.fps)
    if not args.no_ground_align:
        frames = ground_align_frames(frames)

    human_json = out_dir / "references" / "unitree_g1" / "lafan1" / f"{clip}.human_frames.json"
    contacts_json = out_dir / "contacts" / "unitree_g1" / "lafan1" / f"{clip}.contacts.json"
    quality_json = out_dir / "reports" / "unitree_g1" / "lafan1" / f"{clip}.quality.json"
    source_json = out_dir / "source" / "unitree_g1" / "lafan1" / f"{clip}.qpos.json"

    _write_json(
        human_json,
        _reference_payload(
            trajectory,
            frames,
            contacts,
            quality,
            site_map,
            clip=clip,
            mapping=mapping,
            ground_aligned=not args.no_ground_align,
        ),
    )
    _write_json(contacts_json, {"fps": trajectory.fps, "contacts": contacts})
    _write_json(quality_json, asdict(quality))
    _write_json(source_json, _source_qpos_payload(trajectory))
    print(f"[robot-to-gmr] canonical: {human_json}")

    if args.skip_robot_b:
        return 0

    targets = parse_robot_b_list(REPO, args.robot_b)
    for target in targets:
        final_json = out_dir / "robot_b" / target.name / "lafan1" / f"{clip}.qpos.json"
        raw_json = final_json.with_name(f"{clip}.raw.qpos.json")
        batch_output = raw_json if args.postprocess != "none" or args.keep_raw else final_json
        _run_batch_to(
            cpp_cli=args.batch_cli.expanduser().resolve(),
            human_json=human_json,
            out_json=batch_output,
            robot=target.name,
            fast=args.fast,
            verbose=args.verbose,
            parallel=args.parallel,
            contact_ground=not args.no_contact_ground,
            joint_limit_margin_deg=args.joint_limit_margin_deg,
        )

        payload = json.loads(batch_output.read_text(encoding="utf-8"))
        qpos, post_meta = _postprocess(
            np.asarray(payload["qpos_frames"], dtype=np.float64),
            model_xml=target.model_xml,
            contacts=contacts,
            fps=trajectory.fps,
            planar_base=target.planar_base,
            align_wrists=(not args.no_align_wrists) and model_has_wrist_pitch_yaw(target.model_xml),
            mode=args.postprocess,
        )
        payload["qpos_frames"] = qpos.tolist()
        payload["postprocess"] = post_meta
        _write_json(final_json, payload)
        print(f"[robot-to-gmr] {target.name}: {final_json}")
        if "stance_slip_mps" in post_meta:
            slip = post_meta["stance_slip_mps"]
            print(f"[robot-to-gmr] stance slip {slip['before']:.4f} -> {slip['after']:.4f} m/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

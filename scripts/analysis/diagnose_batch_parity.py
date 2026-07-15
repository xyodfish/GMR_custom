#!/usr/bin/env python3
"""Diagnose Py vs C++ batch TO parity: bootstrap, ground_z, contact mask."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.batch_trajectory_retarget import BatchTrajectoryConfig, BatchTrajectoryRetargeter
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
from scripts.analysis.compare_py_vs_cpp_batch import export_human_json, run_cpp_batch, run_python_batch


def main() -> None:
    pt = REPO / "output/gvhmr_pt/cxk-ball_hmr4d_results.pt"
    max_frames = 120
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt, REPO / "assets/body_models"
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    frames = frames[:max_frames]

    kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
        contact_ground=True,
        motion_fps=fps,
    )
    gmr = GMR(**kwargs)
    py_ik = np.stack([gmr.retarget(f).copy() for f in frames])
    print(f"Python ground_z: {gmr.contact_ground.ground_aligner.ground_z}")
    print(f"Python foot_body_ids: {gmr.contact_ground.foot_body_ids}")

    batch = BatchTrajectoryRetargeter(gmr, BatchTrajectoryConfig(verbose=False, show_progress=False))
    py_mask = batch._batch_contact_mask(py_ik)
    py_contact_ratio = float(np.mean(py_mask))
    print(f"Python contact frame ratio: {py_contact_ratio:.3f}")

    human_json = REPO / "output" / "_diag_human.json"
    cpp_out = REPO / "output" / "_diag_cpp.json"
    export_human_json(pt, human_json, max_frames)

    q_py, _, _ = run_python_batch(
        pt, "unitree_g1", max_frames, True, 16, 8, 3, False
    )
    q_cpp, _ = run_cpp_batch(
        human_json,
        "unitree_g1",
        cpp_out,
        max_frames,
        height,
        True,
        16,
        8,
        3,
        False,
        REPO / "cpp/build/gmr_batch_to_cli",
        "best",
    )

    print(f"\nIK bootstrap RMSE py_ik internal: {np.sqrt(np.mean((py_ik - py_ik) ** 2)):.6f}")
    print(f"Full batch RMSE py vs cpp: {np.sqrt(np.mean((q_py - q_cpp) ** 2)):.5f}")
    print(f"Full batch max_abs py vs cpp: {np.max(np.abs(q_py - q_cpp)):.5f}")

    # Per-phase: compare py batch output vs py ik (how much batch changed)
    print(f"Py batch vs py IK RMSE: {np.sqrt(np.mean((q_py - py_ik) ** 2)):.5f}")
    print(f"Cpp batch vs py IK RMSE: {np.sqrt(np.mean((q_cpp - py_ik) ** 2)):.5f}")

    import json

    payload = json.loads(human_json.read_text())
    frames_json = payload["frames"][:max_frames]
    gmr2 = GMR(**kwargs)
    py_ik_json = []
    for fr in frames_json:
        py_ik_json.append(gmr2.retarget(fr).copy())
    py_ik_json = np.stack(py_ik_json)
    print(f"Py IK json vs pt RMSE: {np.sqrt(np.mean((py_ik - py_ik_json) ** 2)):.6f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare Python vs C++ batch window cost at identical q_init."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.batch_trajectory_retarget import (
    BatchTrajectoryConfig,
    BatchTrajectoryRetargeter,
)
from general_motion_retargeting.human_frame_loaders import frame_to_json_dict
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
)


def main() -> None:
    pt = REPO_ROOT / "output/gvhmr_pt/cxk-ball_hmr4d_results.pt"
    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        pt, REPO_ROOT / "assets/body_models"
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    frames = frames[:120]

    kwargs = dict(
        actual_human_height=height,
        src_human="smplx",
        tgt_robot="unitree_g1",
        verbose=False,
        contact_ground=True,
        motion_fps=fps,
    )
    gmr = GMR(**kwargs)
    ik_q = np.stack([gmr.retarget(f).copy() for f in frames])

    batch = BatchTrajectoryRetargeter(
        GMR(**kwargs),
        BatchTrajectoryConfig(window_size=16, window_stride=8, gn_steps=3, verbose=False, show_progress=False),
    )
    batch.set_motion_fps(fps)

    prepared = [batch.gmr._prepare_scaled_human_data(f) for f in frames]
    targets = [batch._targets_for_prepared(p) for p in prepared]

    q_win = ik_q[:16]
    tgt_win = targets[:16]
    anchor = q_win[0].copy()
    batch._window_frame_offset = 0
    batch._window_anchor_w = 0.0
    py_cost = batch._window_cost(q_win, tgt_win, anchor, ik_q[:16])

    fk_only = sum(batch._fk_tracking_cost(q, t) for q, t in zip(q_win, tgt_win))
    foot_only = batch._window_foot_cost(q_win, ik_q[:16])

    print(f"Python window [0:16) cost={py_cost:.4f} fk={fk_only:.4f} foot={foot_only:.4f}")
    print(f"track_entries={len(batch._track_entries)} ground_z={batch._ground_z}")

    # target diff: export first frame targets
    t0 = targets[0]
    for name in sorted(t0.keys()):
        pos, quat = t0[name]
        print(f"  target {name}: pos={pos}")

    # Compare with C++ by exporting q_init and running verbose on window 0 only - use full batch
    payload = {
        "fps": float(fps),
        "actual_human_height": float(height),
        "src_human": "smplx",
        "frames": [frame_to_json_dict(f) for f in frames],
    }
    with tempfile.TemporaryDirectory(prefix="diag_gn_") as tmp:
        human_json = pathlib.Path(tmp) / "human.json"
        out_json = pathlib.Path(tmp) / "out.json"
        qinit_json = pathlib.Path(tmp) / "qinit.json"
        human_json.write_text(json.dumps(payload))
        qinit_json.write_text(json.dumps({"qpos_frames": [row.tolist() for row in ik_q]}))

        cmd = [
            str(REPO_ROOT / "cpp/build/gmr_batch_to_cli"),
            "--gmr_root",
            str(REPO_ROOT),
            "--robot",
            "unitree_g1",
            "--human_frame_json",
            str(human_json),
            "--out_json",
            str(out_json),
            "--max_frames",
            "120",
            "--contact_ground",
            "--verbose",
            "--q_init_json",
            str(qinit_json),
        ]
        import os

        env = os.environ.copy()
        devel_lib = "/opt/robot/devel/lib"
        if pathlib.Path(devel_lib).is_dir():
            env["LD_LIBRARY_PATH"] = f"{devel_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        for line in r.stderr.splitlines():
            if "GN window offset=0" in line or (
                "windowCost breakdown" in line and "offset=0" not in line
            ):
                if "GN window offset=0" in line or line.count("windowCost") <= 2:
                    print("CPP:", line)
        if r.returncode != 0:
            print(r.stderr[-3000:])


if __name__ == "__main__":
    main()

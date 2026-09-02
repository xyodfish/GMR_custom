#!/usr/bin/env python3
"""Retarget a SMPL-X NPZ dataset to training-ready Unitree H2 PKLs.

The H2 retargeter uses the repository's SMPL-X -> G1 -> H2 batch trajectory
optimization path. Outputs preserve the input directory tree and are written
atomically, so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from h2_motion_quality import H2MotionQualityGate, QUALITY_SCHEMA


REPO = Path(__file__).resolve().parents[2]
RUN_CPP = REPO / "scripts" / "tools" / "run_cpp_batch_to.py"
H2_XML = REPO / "assets" / "unitree_h2" / "h2.xml"
G1_XML = REPO / "assets" / "unitree_g1" / "g1.xml"
DEFAULT_INPUT = Path.home() / "Workspace" / "data"
DEFAULT_OUTPUT = Path.home() / "Workspace" / "gmr_cg_batch_h2"
DEFAULT_PYTHON = Path.home() / "miniconda3" / "envs" / "gmr" / "bin" / "python"
EXPECTED_NQ = 36
EXPECTED_DOF = 29


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_qpos_json(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = json.loads(path.read_text())
    qpos = np.asarray(payload["qpos_frames"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] != EXPECTED_NQ:
        raise ValueError(f"expected non-empty H2 qpos [T,{EXPECTED_NQ}], got {qpos.shape}")

    if not np.isfinite(qpos).all():
        raise ValueError("H2 qpos contains NaN or infinity")

    quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(quaternion_norm < 1.0e-6):
        raise ValueError("H2 root quaternion contains a zero-norm frame")

    qpos[:, 3:7] /= quaternion_norm[:, None]
    return payload, qpos


def _h2_local_body_positions(qpos: np.ndarray) -> tuple[np.ndarray, list[str]]:
    model = mujoco.MjModel.from_xml_path(str(H2_XML))
    if model.nq != EXPECTED_NQ:
        raise ValueError(f"H2 XML nq changed: expected {EXPECTED_NQ}, got {model.nq}")

    data = mujoco.MjData(model)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]
    local_body_pos = np.empty((qpos.shape[0], model.nbody - 1, 3), dtype=np.float32)
    for frame_index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        root_rotation = data.xmat[1].reshape(3, 3)
        world_offsets = data.xpos[1:] - data.xpos[1]
        local_body_pos[frame_index] = world_offsets @ root_rotation

    return local_body_pos, body_names


def _training_payload(
    source: Path,
    retarget_payload: dict[str, Any],
    qpos: np.ndarray,
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    local_body_pos, body_names = _h2_local_body_positions(qpos)
    method = str(retarget_payload.get("method", ""))
    if not bool(retarget_payload.get("g1_bridge")) or "via_g1" not in method:
        raise ValueError(f"retarget output did not use the required G1 bridge: method={method!r}")

    qpos_f32 = qpos.astype(np.float32)
    return {
        "fps": float(retarget_payload["fps"]),
        "root_pos": qpos_f32[:, :3].copy(),
        "root_rot": qpos_f32[:, 3:7][:, [1, 2, 3, 0]].copy(),
        "dof_pos": qpos_f32[:, 7:].copy(),
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
        "qpos": qpos_f32,
        "robot": "unitree_h2",
        "method": method,
        "src_human": str(retarget_payload.get("src_human", "smplx")),
        "input_file": str(source.resolve()),
        "g1_bridge": True,
        "g1_bridge_profile": retarget_payload.get("g1_bridge_profile"),
        "profile": retarget_payload.get("profile"),
        "training_schema": "gmr_h2_fk_v2_joint_map",
        "quality_schema": quality_report["quality_schema"],
        "quality_status": quality_report["status"],
        "quality_anomaly_score": quality_report["anomaly_score"],
        "quality_issues": quality_report["issues"],
        "quality_metrics": quality_report["metrics"],
    }


def _valid_training_pkl(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        root_pos = np.asarray(payload["root_pos"])
        root_rot = np.asarray(payload["root_rot"])
        dof_pos = np.asarray(payload["dof_pos"])
        local_body_pos = np.asarray(payload["local_body_pos"])
        body_names = payload["link_body_list"]
        frames = root_pos.shape[0]
        return (
            payload.get("robot") == "unitree_h2"
            and payload.get("g1_bridge") is True
            and payload.get("training_schema") == "gmr_h2_fk_v2_joint_map"
            and payload.get("quality_schema") == QUALITY_SCHEMA
            and payload.get("quality_status") in ("accepted", "quarantine")
            and "joint_map" in str(payload.get("method", ""))
            and root_pos.shape == (frames, 3)
            and root_rot.shape == (frames, 4)
            and dof_pos.shape == (frames, EXPECTED_DOF)
            and local_body_pos.shape == (frames, len(body_names), 3)
            and frames > 0
            and np.isfinite(root_pos).all()
            and np.isfinite(root_rot).all()
            and np.isfinite(dof_pos).all()
            and np.isfinite(local_body_pos).all()
        )
    except (EOFError, KeyError, OSError, pickle.PickleError, TypeError, ValueError):
        return False


def _retarget_one(
    source: Path,
    *,
    python: Path,
    body_model_dir: Path,
    quality: bool,
    quality_gate: H2MotionQualityGate,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="gmr_h2_dataset_") as temporary_dir:
        qpos_json = Path(temporary_dir) / "h2.qpos.json"
        g1_qpos_json = Path(temporary_dir) / "g1.qpos.json"
        command = [
            str(python),
            str(RUN_CPP),
            "--input_file",
            str(source),
            "--input_type",
            "smplx",
            "--robot",
            "unitree_h2",
            "--body_model_dir",
            str(body_model_dir),
            "--out_json",
            str(qpos_json),
            "--backend",
            "mujoco_se3",
            "--contact_ground",
            "--dump_g1_bridge_json",
            str(g1_qpos_json),
            "--quality" if quality else "--fast",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = f"{REPO}:{environment.get('PYTHONPATH', '')}"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            output_tail = "\n".join(result.stdout.splitlines()[-80:])
            raise RuntimeError(f"retarget command exited {result.returncode}\n{output_tail}")

        retarget_payload, qpos = _load_qpos_json(qpos_json)
        _, g1_qpos = _load_qpos_json(g1_qpos_json)
        quality_report = quality_gate.evaluate(
            qpos,
            g1_qpos,
            float(retarget_payload["fps"]),
            retarget_payload["g1_tracking_quality"],
        )
        motion = _training_payload(source, retarget_payload, qpos, quality_report)
        return motion, quality_report, "\n".join(result.stdout.splitlines()[-8:])


def _output_path(source: Path, input_root: Path, output_root: Path, category: str) -> Path:
    relative = source.relative_to(input_root)
    return output_root / category / relative.parent / f"{relative.stem}_gmr.pkl"


def _quality_record_path(source: Path, input_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(input_root)
    return output_root / "quality" / "records" / relative.parent / f"{relative.stem}.quality.json"


def _atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _write_quality_indexes(output_root: Path) -> None:
    records = []
    records_root = output_root / "quality" / "records"
    if records_root.is_dir():
        for path in records_root.rglob("*.quality.json"):
            records.append(json.loads(path.read_text()))

    records.sort(key=lambda item: (-float(item["anomaly_score"]), item["input_file"]))
    accepted = sorted(item["output_file"] for item in records if item["status"] == "accepted")
    quarantine = sorted(item["output_file"] for item in records if item["status"] == "quarantine")
    quality_root = output_root / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        quality_root / "ranking.json",
        {
            "quality_schema": QUALITY_SCHEMA,
            "generated_at": _utc_now(),
            "total": len(records),
            "accepted": len(accepted),
            "quarantine": len(quarantine),
            "records": records,
        },
    )
    (quality_root / "accepted.txt").write_text("".join(f"{path}\n" for path in accepted))
    (quality_root / "quarantine.txt").write_text("".join(f"{path}\n" for path in quarantine))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-glob", default="*_stageii.npz")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--body-model-dir", type=Path, default=REPO / "assets" / "body_models")
    parser.add_argument("--quality", action="store_true", help="Use the slower offline quality preset.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new clips; zero means all.")
    parser.add_argument(
        "--index-interval",
        type=int,
        default=100,
        help="Refresh ranked quality indexes after this many generated clips.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    body_model_dir = args.body_model_dir.expanduser().resolve()
    for required in (input_root, python, body_model_dir, RUN_CPP, H2_XML, G1_XML):
        if not required.exists():
            parser.error(f"required path does not exist: {required}")

    if args.limit < 0:
        parser.error("--limit must be non-negative")

    if args.index_interval <= 0:
        parser.error("--index-interval must be positive")

    sources = sorted(input_root.rglob(args.input_glob))
    if not sources:
        parser.error(f"no files matched {args.input_glob!r} under {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "retarget_manifest.json"
    failure_path = output_root / "retarget_failures.jsonl"
    quality_event_path = output_root / "quality" / "events.jsonl"
    quality_event_path.parent.mkdir(parents=True, exist_ok=True)
    quality_gate = H2MotionQualityGate(H2_XML, G1_XML)
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    completed = 0
    skipped = 0
    failed = 0
    attempted = 0
    accepted = 0
    quarantined = 0
    warning_clips = 0

    def write_manifest(current_source: Path | None = None) -> None:
        _atomic_json(
            manifest_path,
            {
                "status": "running",
                "pid": os.getpid(),
                "started_at": started_at,
                "updated_at": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                "input_root": str(input_root),
                "output_root": str(output_root),
                "input_glob": args.input_glob,
                "total_inputs": len(sources),
                "completed_this_run": completed,
                "skipped_valid": skipped,
                "failed_this_run": failed,
                "accepted_this_run": accepted,
                "quarantined_this_run": quarantined,
                "warning_clips_this_run": warning_clips,
                "current_source": str(current_source) if current_source else None,
                "preset": "quality" if args.quality else "fast",
                "method": "smplx_to_h2_via_g1_joint_map",
                "quality_schema": QUALITY_SCHEMA,
                "training_data_root": str(output_root / "accepted"),
                "quarantine_root": str(output_root / "quarantine"),
            },
        )

    print(
        f"[h2-dataset] inputs={len(sources)} preset={'quality' if args.quality else 'fast'} "
        f"input={input_root} output={output_root}",
        flush=True,
    )
    write_manifest()

    def mark_stopped() -> None:
        try:
            state = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            return

        if state.get("pid") != os.getpid() or state.get("status") != "running":
            return

        state.update(
            {
                "status": "stopped",
                "stopped_at": _utc_now(),
                "updated_at": _utc_now(),
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                "current_source": None,
            }
        )
        _atomic_json(manifest_path, state)
        _write_quality_indexes(output_root)

    atexit.register(mark_stopped)
    for index, source in enumerate(sources, start=1):
        accepted_destination = _output_path(source, input_root, output_root, "accepted")
        quarantine_destination = _output_path(source, input_root, output_root, "quarantine")
        record_path = _quality_record_path(source, input_root, output_root)
        existing_destination = next(
            (
                path
                for path in (accepted_destination, quarantine_destination)
                if path.is_file() and _valid_training_pkl(path)
            ),
            None,
        )
        if existing_destination is not None and record_path.is_file():
            skipped += 1
            continue

        if args.limit and attempted >= args.limit:
            break

        attempted += 1
        write_manifest(source)
        clip_started = time.monotonic()
        print(f"[h2-dataset] [{index}/{len(sources)}] {source}", flush=True)
        try:
            motion, quality_report, log_tail = _retarget_one(
                source,
                python=python,
                body_model_dir=body_model_dir,
                quality=args.quality,
                quality_gate=quality_gate,
            )
            category = str(quality_report["status"])
            destination = accepted_destination if category == "accepted" else quarantine_destination
            record = {
                **quality_report,
                "generated_at": _utc_now(),
                "input_file": str(source),
                "output_file": str(destination),
                "relative_input": str(source.relative_to(input_root)),
                "relative_output": str(destination.relative_to(output_root)),
            }
            _atomic_pickle(destination, motion)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(record_path, record)
            with quality_event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            completed += 1
            accepted += category == "accepted"
            quarantined += category == "quarantine"
            warning_clips += any(issue["severity"] == "warning" for issue in quality_report["issues"])
            elapsed = time.monotonic() - clip_started
            issue_codes = ",".join(issue["code"] for issue in quality_report["issues"]) or "none"
            print(
                f"[h2-dataset] {category.upper()} pkl={destination} "
                f"issues={issue_codes} elapsed={elapsed:.2f}s",
                flush=True,
            )
            if log_tail:
                print(log_tail, flush=True)

            if completed % args.index_interval == 0:
                _write_quality_indexes(output_root)
        except Exception as error:
            failed += 1
            failure = {
                "time": _utc_now(),
                "input_file": str(source),
                "accepted_output_file": str(accepted_destination),
                "quarantine_output_file": str(quarantine_destination),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            with failure_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(f"[h2-dataset] FAIL {source}: {type(error).__name__}: {error}", flush=True)
            if args.stop_on_error:
                raise

        write_manifest()

    final_status = "complete" if attempted + skipped >= len(sources) else "stopped_at_limit"
    final_manifest = json.loads(manifest_path.read_text())
    final_manifest.update(
        {
            "status": final_status,
            "finished_at": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            "completed_this_run": completed,
            "skipped_valid": skipped,
            "failed_this_run": failed,
            "accepted_this_run": accepted,
            "quarantined_this_run": quarantined,
            "warning_clips_this_run": warning_clips,
            "current_source": None,
        }
    )
    _atomic_json(manifest_path, final_manifest)
    _write_quality_indexes(output_root)
    atexit.unregister(mark_stopped)
    print(
        f"[h2-dataset] status={final_status} completed={completed} accepted={accepted} "
        f"quarantined={quarantined} skipped={skipped} failed={failed}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

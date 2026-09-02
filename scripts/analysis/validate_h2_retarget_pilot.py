#!/usr/bin/env python3
"""Compare the mapped G1 baseline with H2 Batch TO on a small, explicit pilot set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUN_CPP = REPO / "scripts" / "tools" / "run_cpp_batch_to.py"
H2_XML = REPO / "assets" / "unitree_h2" / "h2.xml"
G1_XML = REPO / "assets" / "unitree_g1" / "g1.xml"
DEFAULT_PYTHON = Path.home() / "miniconda3" / "envs" / "gmr" / "bin" / "python"


def _load_quality_tools() -> tuple[type, Any]:
    import sys

    retarget_scripts = REPO / "scripts" / "retarget"
    sys.path.insert(0, str(retarget_scripts))
    from h2_motion_quality import H2MotionQualityGate, material_quality_regressions

    return H2MotionQualityGate, material_quality_regressions


def _run(
    source: Path,
    output: Path,
    g1_output: Path,
    *,
    python: Path,
    body_model_dir: Path,
    max_frames: int,
    gn_steps: int,
    w_reference: float,
) -> dict[str, Any]:
    command = [
        str(python),
        str(RUN_CPP),
        "--input_file",
        str(source),
        "--input_type",
        "smplx",
        "--robot",
        "unitree_h2",
        "--out_json",
        str(output),
        "--body_model_dir",
        str(body_model_dir),
        "--contact_ground",
        "--quality",
        "--h2_refine",
        "--gn_steps",
        str(gn_steps),
        "--w_reference",
        str(w_reference),
        "--dump_g1_bridge_json",
        str(g1_output),
    ]
    if max_frames > 0:
        command += ["--max_frames", str(max_frames)]

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
        tail = "\n".join(result.stdout.splitlines()[-60:])
        raise RuntimeError(f"retarget failed for {source}\n{tail}")

    return json.loads(output.read_text())


def _selected_metrics(report: dict[str, Any], tracking: dict[str, Any]) -> dict[str, float]:
    metrics = report["metrics"]
    return {
        "position_p95_m": float(tracking["position_p95_m"]),
        "position_max_m": float(tracking["position_max_m"]),
        "rotation_p95_deg": float(tracking["rotation_p95_deg"]),
        "rotation_max_deg": float(tracking["rotation_max_deg"]),
        "joint_step_max_rad": float(metrics["joint_step_max_rad"]),
        "joint_acceleration_max_rad_s2": float(metrics["joint_acceleration_max_rad_s2"]),
        "foot_slip_p95_m_s": float(metrics["contact_foot_slip_p95_m_s"]),
        "foot_penetration_max_m": float(metrics["foot_ground_penetration_max_m"]),
        "protected_penetration_max_m": float(
            metrics["protected_body_ground_penetration_max_m"]
        ),
    }


def _median(rows: list[dict[str, Any]], side: str, metric: str) -> float:
    return float(np.median([row[side]["metrics"][metric] for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--inputs-file", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--body-model-dir", type=Path, default=REPO / "assets" / "body_models")
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--candidate-gn-steps", type=int, default=4)
    parser.add_argument("--w-reference", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    inputs = [path.expanduser().resolve() for path in args.inputs]
    if args.inputs_file is not None:
        listed_inputs = json.loads(args.inputs_file.expanduser().read_text())
        if not isinstance(listed_inputs, list) or not all(
            isinstance(path, str) for path in listed_inputs
        ):
            parser.error("--inputs-file must contain a JSON array of paths")

        inputs.extend(Path(path).expanduser().resolve() for path in listed_inputs)

    inputs = list(dict.fromkeys(inputs))
    if not inputs:
        parser.error("provide at least one input or --inputs-file")

    required = [*inputs, args.python.expanduser(), args.body_model_dir.expanduser(), RUN_CPP]
    missing = [path for path in required if not path.exists()]
    if missing:
        parser.error("missing required paths: " + ", ".join(str(path) for path in missing))

    if args.max_frames < 0 or args.candidate_gn_steps <= 0:
        parser.error("--max-frames must be non-negative and --candidate-gn-steps must be positive")

    if args.w_reference < 0.0:
        parser.error("--w-reference must be non-negative")

    QualityGate, compare_quality = _load_quality_tools()
    gate = QualityGate(H2_XML, G1_XML)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gmr_h2_pilot_") as temporary:
        temporary_root = Path(temporary)
        for index, source in enumerate(inputs):
            baseline_path = temporary_root / f"{index}_baseline.json"
            baseline_g1_path = temporary_root / f"{index}_baseline_g1.json"
            candidate_path = temporary_root / f"{index}_candidate.json"
            candidate_g1_path = temporary_root / f"{index}_candidate_g1.json"
            baseline = _run(
                source,
                baseline_path,
                baseline_g1_path,
                python=args.python.expanduser(),
                body_model_dir=args.body_model_dir.expanduser(),
                max_frames=args.max_frames,
                gn_steps=0,
                w_reference=args.w_reference,
            )
            candidate = _run(
                source,
                candidate_path,
                candidate_g1_path,
                python=args.python.expanduser(),
                body_model_dir=args.body_model_dir.expanduser(),
                max_frames=args.max_frames,
                gn_steps=args.candidate_gn_steps,
                w_reference=args.w_reference,
            )
            g1 = json.loads(baseline_g1_path.read_text())
            g1_qpos = np.asarray(g1["qpos_frames"], dtype=np.float64)
            baseline_report = gate.evaluate(
                np.asarray(baseline["qpos_frames"], dtype=np.float64),
                g1_qpos,
                float(baseline["fps"]),
                baseline["tracking_quality"],
                require_bridge_identity=False,
            )
            candidate_report = gate.evaluate(
                np.asarray(candidate["qpos_frames"], dtype=np.float64),
                g1_qpos,
                float(candidate["fps"]),
                candidate["tracking_quality"],
                require_bridge_identity=False,
            )
            rows.append(
                {
                    "input": str(source),
                    "baseline": {
                        "status": baseline_report["status"],
                        "issues": baseline_report["issues"],
                        "metrics": _selected_metrics(
                            baseline_report,
                            baseline["tracking_quality"],
                        ),
                    },
                    "candidate": {
                        "status": candidate_report["status"],
                        "issues": candidate_report["issues"],
                        "metrics": _selected_metrics(
                            candidate_report,
                            candidate["tracking_quality"],
                        ),
                    },
                }
            )
            print(f"[h2-pilot] {index + 1}/{len(inputs)} {source.name}", flush=True)

    median_metrics = {}
    for metric in rows[0]["baseline"]["metrics"]:
        baseline_median = _median(rows, "baseline", metric)
        candidate_median = _median(rows, "candidate", metric)
        median_metrics[metric] = {
            "baseline": baseline_median,
            "candidate": candidate_median,
            "ratio": candidate_median / baseline_median if baseline_median > 1.0e-12 else 0.0,
        }

    new_rejects = []
    per_clip_regressions = []
    for row in rows:
        baseline_codes = {
            issue["code"] for issue in row["baseline"]["issues"] if issue["severity"] == "reject"
        }
        candidate_codes = {
            issue["code"] for issue in row["candidate"]["issues"] if issue["severity"] == "reject"
        }
        if candidate_codes - baseline_codes:
            new_rejects.append(
                {"input": row["input"], "new_reject_codes": sorted(candidate_codes - baseline_codes)}
            )

        baseline_for_compare = {
            "issues": row["baseline"]["issues"],
            "metrics": {
                **row["baseline"]["metrics"],
                "contact_foot_slip_p95_m_s": row["baseline"]["metrics"]["foot_slip_p95_m_s"],
                "g1_semantic_tracking": {
                    "position_p95_m": row["baseline"]["metrics"]["position_p95_m"],
                    "rotation_p95_deg": row["baseline"]["metrics"]["rotation_p95_deg"],
                },
            },
        }
        candidate_for_compare = {
            "issues": row["candidate"]["issues"],
            "metrics": {
                **row["candidate"]["metrics"],
                "contact_foot_slip_p95_m_s": row["candidate"]["metrics"]["foot_slip_p95_m_s"],
                "g1_semantic_tracking": {
                    "position_p95_m": row["candidate"]["metrics"]["position_p95_m"],
                    "rotation_p95_deg": row["candidate"]["metrics"]["rotation_p95_deg"],
                },
            },
        }
        regressed = [
            regression
            for regression in compare_quality(baseline_for_compare, candidate_for_compare)
            if regression["metric"] != "new_reject_codes"
        ]

        if regressed:
            per_clip_regressions.append({"input": row["input"], "metrics": regressed})

    checks = {
        "no_new_rejects": not new_rejects,
        "no_material_per_clip_regression": not per_clip_regressions,
        "position_p95_not_worse": median_metrics["position_p95_m"]["ratio"] <= 1.0,
        "rotation_p95_not_worse": median_metrics["rotation_p95_deg"]["ratio"] <= 1.05,
        "acceleration_not_worse": median_metrics["joint_acceleration_max_rad_s2"]["ratio"] <= 1.10,
        "foot_slip_not_worse": median_metrics["foot_slip_p95_m_s"]["ratio"] <= 1.10,
        "no_foot_penetration_reject": max(
            row["candidate"]["metrics"]["foot_penetration_max_m"] for row in rows
        ) <= 0.02,
        "no_protected_penetration_reject": max(
            row["candidate"]["metrics"]["protected_penetration_max_m"] for row in rows
        ) <= 0.03,
    }
    passed = all(checks.values())
    output = {
        "schema": "gmr_h2_pilot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_frames": args.max_frames,
            "candidate_gn_steps": args.candidate_gn_steps,
            "w_reference": args.w_reference,
        },
        "passed": passed,
        "checks": checks,
        "new_rejects": new_rejects,
        "per_clip_regressions": per_clip_regressions,
        "median_metrics": median_metrics,
        "clips": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"[h2-pilot] {'PASS' if passed else 'FAIL'} report={args.out}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

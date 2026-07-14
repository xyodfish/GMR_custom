import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


REQUIRED_MODEL_FILES = (
    "SMPLX_NEUTRAL.pkl",
    "SMPLX_FEMALE.pkl",
    "SMPLX_MALE.pkl",
)


def import_smpl_helpers():
    try:
        from general_motion_retargeting.utils.smpl import (
            get_smplx_data_offline_fast,
            load_smplx_file,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name
        print(f"[ERROR] Missing Python dependency: {missing}")
        print("Install the project dependencies first, for example:")
        print("  pip install -e .")
        if missing == "torch":
            print("If torch is still missing, install a torch build that matches your CUDA/CPU setup.")
        raise SystemExit(1) from exc

    return load_smplx_file, get_smplx_data_offline_fast


def import_smplx_runtime():
    missing = []
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
        missing.append("torch")

    try:
        import smplx
    except ModuleNotFoundError:
        smplx = None
        missing.append("smplx")

    if missing:
        print("[ERROR] Missing Python dependency/dependencies: " + ", ".join(missing))
        print("Install the project dependencies first, for example:")
        print("  pip install -e .")
        print("If torch is still missing, install a torch build that matches your CUDA/CPU setup.")
        raise SystemExit(1)

    return torch, smplx


def check_body_models(body_model_dir):
    smplx_dir = body_model_dir / "smplx"
    if not smplx_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {smplx_dir}")

    missing = [name for name in REQUIRED_MODEL_FILES if not (smplx_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing SMPL-X pkl model file(s): "
            + ", ".join(missing)
            + f"\nExpected under: {smplx_dir}"
        )

    return smplx_dir


def write_tiny_smplx_motion(path, gender, frames, fps):
    pose_body = np.zeros((frames, 63), dtype=np.float32)
    root_orient = np.zeros((frames, 3), dtype=np.float32)
    trans = np.zeros((frames, 3), dtype=np.float32)

    # Add tiny motion so the output is not completely static.
    trans[:, 0] = np.linspace(0.0, 0.05, frames, dtype=np.float32)
    root_orient[:, 2] = np.linspace(0.0, 0.08, frames, dtype=np.float32)

    np.savez(
        path,
        gender=np.array(gender),
        pose_body=pose_body,
        betas=np.zeros(16, dtype=np.float32),
        root_orient=root_orient,
        trans=trans,
        mocap_frame_rate=np.array(fps),
    )


def run_optional_retarget(frame_data, actual_human_height, robot, motion_fps):
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        motion_fps=motion_fps,
    )
    qpos = retarget.retarget(frame_data)
    print(f"[OK] Retargeted one frame to {robot}: qpos shape={qpos.shape}")
    print(f"[OK] qpos[:8]={np.array2string(qpos[:8], precision=4)}")


def run_direct_smplx_demo(body_model_dir, gender, frames):
    torch, smplx = import_smplx_runtime()

    body_model = smplx.create(
        model_path=str(body_model_dir),
        model_type="smplx",
        gender=gender,
        ext="pkl",
        use_pca=False,
        batch_size=frames,
    )

    with torch.no_grad():
        output = body_model(
            betas=torch.zeros(1, 16).float(),
            global_orient=torch.zeros(frames, 3).float(),
            body_pose=torch.zeros(frames, 63).float(),
            transl=torch.zeros(frames, 3).float(),
            left_hand_pose=torch.zeros(frames, 45).float(),
            right_hand_pose=torch.zeros(frames, 45).float(),
            jaw_pose=torch.zeros(frames, 3).float(),
            leye_pose=torch.zeros(frames, 3).float(),
            reye_pose=torch.zeros(frames, 3).float(),
            return_full_pose=True,
        )

    pelvis = output.joints[0, 0].detach().cpu().numpy()
    print(f"[OK] Direct smplx.create loaded gender={gender}")
    print(f"[OK] vertices shape={tuple(output.vertices.shape)}")
    print(f"[OK] joints shape={tuple(output.joints.shape)}")
    print(f"[OK] pelvis joint={np.array2string(pelvis, precision=4)}")
    return body_model


def main():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    parser = argparse.ArgumentParser(
        description="Run a tiny SMPL-X Python sanity demo using local body models."
    )
    parser.add_argument(
        "--body_model_dir",
        type=Path,
        default=repo_root / "assets" / "body_models",
        help="Folder containing the smplx subfolder.",
    )
    parser.add_argument(
        "--gender",
        choices=("neutral", "female", "male"),
        default="neutral",
        help="SMPL-X body model gender to load.",
    )
    parser.add_argument("--frames", type=int, default=8, help="Number of synthetic frames.")
    parser.add_argument("--fps", type=int, default=30, help="Synthetic motion FPS.")
    parser.add_argument(
        "--project-pipeline",
        action="store_true",
        help="Also run the repository load_smplx_file/get_smplx_data_offline_fast path.",
    )
    parser.add_argument(
        "--retarget",
        metavar="ROBOT",
        default=None,
        help="Optionally run one GMR retarget frame, e.g. --retarget unitree_g1.",
    )
    args = parser.parse_args()

    if args.frames < 2:
        raise SystemExit("[ERROR] --frames must be >= 2")

    body_model_dir = args.body_model_dir.expanduser().resolve()
    smplx_dir = check_body_models(body_model_dir)
    print(f"[OK] Found SMPL-X models under: {smplx_dir}")

    run_direct_smplx_demo(body_model_dir, args.gender, args.frames)

    if not args.project_pipeline and not args.retarget:
        print("[OK] SMPL-X Python demo finished successfully.")
        return

    load_smplx_file, get_smplx_data_offline_fast = import_smpl_helpers()

    with tempfile.TemporaryDirectory(prefix="gmr_smplx_demo_") as tmpdir:
        motion_path = Path(tmpdir) / "tiny_smplx_motion.npz"
        write_tiny_smplx_motion(motion_path, args.gender, args.frames, args.fps)
        print(f"[OK] Wrote synthetic motion: {motion_path}")

        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
            motion_path, body_model_dir
        )
        frame_data, aligned_fps = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=args.fps
        )

    first_frame = frame_data[0]
    hips_pos, hips_quat = first_frame["pelvis"]

    print(f"[OK] Loaded body model with {len(body_model.parents)} joints")
    print(f"[OK] Produced {len(frame_data)} retargeting frames at {aligned_fps:.2f} fps")
    print(f"[OK] Estimated human height: {float(actual_human_height):.4f} m")
    print(f"[OK] pelvis pos={np.array2string(hips_pos, precision=4)}")
    print(f"[OK] pelvis quat(wxyz)={np.array2string(hips_quat, precision=4)}")

    if args.retarget:
        run_optional_retarget(first_frame, actual_human_height, args.retarget, aligned_fps)

    print("[OK] SMPL-X Python demo finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise

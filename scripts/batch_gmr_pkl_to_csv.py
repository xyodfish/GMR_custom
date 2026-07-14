import argparse
import pickle
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--ground-feet",
        action="store_true",
        help="Remove vertical root drift by keeping the lower G1 ankle at its ground-contact height.",
    )
    args = parser.parse_args()

    if args.ground_feet:
        import mujoco

        model_path = os.path.join(os.path.dirname(__file__), "..", "assets", "unitree_g1", "g1_mocap_29dof.xml")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        foot_body_ids = [model.body("left_ankle_roll_link").id, model.body("right_ankle_roll_link").id]

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    for i, file in enumerate(os.listdir(args.folder)):
        if file.endswith(".pkl"):
            with open(os.path.join(args.folder, file), "rb") as f:
                motion_data = pickle.load(f)
        else:
            continue

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]            
        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
        motion[:, :3] = motion_data["root_pos"]
        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos

        if args.ground_feet:
            lower_foot_heights = np.empty(motion.shape[0], dtype=np.float32)
            for frame_idx, frame in enumerate(motion):
                data.qpos[:3] = frame[:3]
                data.qpos[3:7] = frame[3:7][[3, 0, 1, 2]]
                data.qpos[7:] = frame[7:]
                mujoco.mj_forward(model, data)
                lower_foot_heights[frame_idx] = min(data.xpos[foot_body_ids, 2])

            contact_height = np.percentile(lower_foot_heights, 5)
            root_z_correction = contact_height - lower_foot_heights
            motion[:, 2] += root_z_correction
            print(
                f"Grounded feet in {file}: root z correction "
                f"[{root_z_correction.min():.3f}, {root_z_correction.max():.3f}] m"
            )
        
        if frame_rate > 30:
            # downsample to 30 fps
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")
        

        np.savetxt(
            os.path.join(args.folder, "csv", file.replace(".pkl", ".csv")),
            motion,
            delimiter=",",
        )
        print(f"({i}/{len(os.listdir(args.folder))}) Saved to {os.path.join(args.folder, 'csv', file.replace('.pkl', '.csv'))}")

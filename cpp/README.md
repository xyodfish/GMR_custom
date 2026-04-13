# GMR C++ Retargeting (Experimental)

This directory is a C++ baseline for GMR retargeting.

## What is implemented
- Backend-agnostic `Retargeter` base class that outputs target joint coordinates (`qpos` stream).
- Two concrete retarget backends:
  - `PinocchioRetargetBackend`
  - `MujocoRetargetBackend`
- Backend selection is independent from renderer target (MuJoCo / ROS / other GUI).
- Reuse of optimization-style QP solver structure from `whole_body_control`
  - `qp_solver` / `hqp_solver` / `qp_data`
- IK config reuse from existing `general_motion_retargeting/ik_configs/*.json`
- A CLI for single-frame retargeting: `gmr_retarget_cli`
- A MuJoCo viewer with YAML runtime config: `gmr_retarget_viewer` (render-only)

## Dependencies
Expected prefix (default):
- `/opt/galbot/devel/x86_64-Linux-GNU-9.4.0`

Required packages:
- `Eigen3`
- `qpOASES`
- `pinocchio`
- `mujoco`
- `nlohmann_json` header (`nlohmann/json.hpp`)
- `yaml-cpp` (`yaml-cpp/yaml.h`)

## Build
```bash
cd /data/open_src_code/GMR_custom
cmake -S cpp -B cpp/build \
  -DGMR_THIRDPARTY_PREFIX=/opt/galbot/devel/x86_64-Linux-GNU-9.4.0 \
  -DGMR_MUJOCO_PREFIX=/opt/galbot/devel_control/x86_64-Linux-GNU-9.4.0
cmake --build cpp/build -j
```

## Run retarget and print/save qpos
```bash
/data/open_src_code/GMR_custom/cpp/build/gmr_retarget_cli \
  --backend pin_ik \
  --gmr_root /data/open_src_code/GMR_custom \
  --robot unitree_g1 \
  --human_frame_json /data/open_src_code/GMR_custom/cpp/examples/human_frame_smplx_g1_example.json \
  --actual_human_height 1.7 \
  --damping 0.5 \
  --max_iter 10 \
  --use_velocity_limit \
  --out_json /data/open_src_code/GMR_custom/tmp/gmr_cpp_qpos.json
```

## Run viewer with YAML config (default realtime)
```bash
/data/open_src_code/GMR_custom/cpp/build/gmr_retarget_viewer \
  --backend pin_ik \
  --config /data/open_src_code/GMR_custom/cpp/examples/retarget_viewer_config.yaml
```

Backend names:
- `pin_ik` (aliases: `pinocchio`, `pinocchio_ik`)
- `mujoco_se3` (aliases: `mujoco`, `se3`)
- `mujoco_jacobian_legacy` (aliases: `mujoco_legacy`, `legacy`)

Command-line options override YAML values, for example:
```bash
/data/open_src_code/GMR_custom/cpp/build/gmr_retarget_viewer \
  --config /data/open_src_code/GMR_custom/cpp/examples/retarget_viewer_config.yaml \
  --precompute
```

## Current scope
- CLI uses the first frame in a multi-frame JSON.
- Current target domain is SMPL-X style human body names (matching existing IK json).
- This is the first drop intended for iterative development (batch mode / pybind / parity tests can be added next).

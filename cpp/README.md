# GMR C++ Retargeting (Experimental)

This directory is a C++ baseline for GMR retargeting.

## What is implemented
- MuJoCo-based full-body IK retarget loop in C++ (`MujocoRetargeter`)
- Reuse of optimization-style QP solver structure from `whole_body_control`
  - `qp_solver` / `hqp_solver` / `qp_data`
- IK config reuse from existing `general_motion_retargeting/ik_configs/*.json`
- A CLI for single-frame retargeting: `gmr_retarget_cli`

## Dependencies
Expected prefix (default):
- `/opt/galbot/devel/x86_64-Linux-GNU-9.4.0`

Required packages:
- `Eigen3`
- `qpOASES`
- `mujoco`
- `nlohmann_json` header (`nlohmann/json.hpp`)

## Build
```bash
cd /data/open_src_code/GMR
cmake -S cpp -B cpp/build \
  -DGMR_THIRDPARTY_PREFIX=/opt/galbot/devel/x86_64-Linux-GNU-9.4.0
cmake --build cpp/build -j
```

## Run retarget and print/save qpos
```bash
/data/open_src_code/GMR/cpp/build/gmr_retarget_cli \
  --gmr_root /data/open_src_code/GMR \
  --robot unitree_g1 \
  --human_frame_json /data/open_src_code/GMR/cpp/examples/human_frame_smplx_g1_example.json \
  --actual_human_height 1.7 \
  --damping 0.5 \
  --max_iter 10 \
  --use_velocity_limit \
  --out_json /data/open_src_code/GMR/tmp/gmr_cpp_qpos.json
```

## Run realtime viewer (retarget + MuJoCo rendering)
```bash
/data/open_src_code/GMR/cpp/build/gmr_retarget_viewer \
  --gmr_root /data/open_src_code/GMR \
  --robot unitree_g1 \
  --human_frame_json /data/open_src_code/GMR/cpp/examples/human_frame_smplx_g1_example.json \
  --actual_human_height 1.7 \
  --damping 0.5 \
  --max_iter 10 \
  --use_velocity_limit \
  --loop
```

## Current scope
- CLI uses the first frame in a multi-frame JSON.
- Current target domain is SMPL-X style human body names (matching existing IK json).
- This is the first drop intended for iterative development (batch mode / pybind / parity tests can be added next).

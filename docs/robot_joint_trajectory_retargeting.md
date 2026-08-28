# Robot joint trajectory to robot joint trajectory

`gmr_robot_to_robot_cli` converts a Unitree G1 joint trajectory into a target robot joint trajectory entirely in C++. It does not start Python and does not require an intermediate HumanFrame JSON. The Python script remains available as a parity reference.

The current source adapter accepts:

- LAFAN1-style G1 CSV (`xyz + xyzw + 29 joints`)
- G1 qpos JSON with `qpos_frames`
- NumPy `[T, 36]` NPY

The pipeline is:

```text
G1 qpos
  -> MuJoCo FK semantic sites
  -> fixed-length canonical SMPL-X proxy + foot contacts
  -> target-specific C++ solver
       humanoid: sliding-window Batch TO
       mobile dual-arm: planar base + staged torso/head/arm IK
  -> stance XY planting
  -> stance-foot orientation IK
  -> contact-aware root-Z ground alignment
  -> optional wrist alignment
  -> target robot qpos JSON
```

## Build

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j --target gmr_robot_to_robot_cli
```

## G1 to H2

For normal interactive use, open `scripts/viz/g1_robot_compare_gui.py` from the IDE. In the browser, select a G1 trajectory and Robot B, then click **纯 C++ 转换并播放**. The page runs the C++ pipeline, registers the result, and opens the side-by-side viewer automatically.

The CLI below is retained for batch jobs and debugging:

```bash
./cpp/build/gmr_robot_to_robot_cli \
  --gmr_root . \
  --input /path/to/g1_motion.qpos.json \
  --robot_b unitree_h2 \
  --out_json output/robot_to_gmr/h2_motion.qpos.json \
  --fast
```

The same command accepts LAFAN1 G1 CSV and C-order little-endian float32/float64 NPY files. Override the default source mapping with `--mapping`.

To inspect the canonical proxy and contact schedule without making it part of the runtime pipeline:

```bash
./cpp/build/gmr_robot_to_robot_cli \
  --gmr_root . \
  --input /path/to/g1_motion.csv \
  --robot_b unitree_h2 \
  --out_json /tmp/h2.qpos.json \
  --dump_human_json /tmp/g1_canonical.human.json
```

Use `--postprocess none` to return the direct Batch TO result. Other useful parity/debug options are `--max_frames`, `--fps`, `--no_ground_align`, `--no_contact_ground`, and `--no_align_wrists`.

The previous Python entry point is retained for regression comparison and multi-target batch orchestration. When using it, run it in the `gmr` environment:

```bash
conda run --no-capture-output -n gmr python \
  scripts/retarget/robot_trajectory_to_gmr_reference.py \
  --input /path/to/g1_motion.qpos.json \
  --robot-b unitree_h2
```

## Outputs

The C++ CLI writes one target qpos JSON to `--out_json`. It includes canonical fit quality, solver profiling, and before/after stance-slip metrics. `--dump_human_json` optionally writes the canonical frames and contact schedule for debugging.

For `galbot_one_golf` and other robots configured with `mobile_upper_body`, the C++ path follows the same semantic protocol as the Python retargeter: planar base motion is copied from the source pelvis, torso and optional head targets are solved first, then both arms reconstruct the human upper-arm and forearm directions using the target robot's own segment lengths. Planar-base DoFs stay frozen during IK and revolute joints retain the configured limit margin.

## Source mapping

The default G1 adapter is:

```text
config/retarget/source/unitree_g1_to_smplx_proxy.yaml
```

It defines the source MJCF, joint order, semantic-site offsets, canonical proportions, smoothing, and contact thresholds. Supporting another source robot requires another adapter and a trajectory reader for that robot's qpos layout.

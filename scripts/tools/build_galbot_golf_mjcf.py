#!/usr/bin/env python3
"""Build GMR MuJoCo model for Galbot One Golf (mobile base + leg + dual arms).

Uses the sphere URDF for kinematics, then swaps primitive geoms for visual meshes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco as mj

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_GALBOT_DESC = Path(
    "/home/xiayu/Workspace/gone/singorix_omnilink/config/galbot_description/"
    "galbot_one_golf_description"
)
DEFAULT_URDF = DEFAULT_GALBOT_DESC / "predefined_description" / "galbot_one_golf_sphere.urdf"
DEFAULT_MESH_SRC = DEFAULT_GALBOT_DESC / "meshes" / "visual"
DEFAULT_OUT = REPO_ROOT / "assets" / "galbot_one_golf" / "galbot_one_golf.xml"

# body name -> mesh path relative to assets/galbot_one_golf/
# leg_link2.obj ships with degenerate line-only sub-objects that break MuJoCo loading.
LEG_LINK2_SKIP_OBJECTS = frozenset({"New_object", "New_object.001"})

BODY_MESHES: dict[str, str] = {
    "base_link": "meshes/visual/chassis/omni_chassis_base_link/obj/omni_chassis_base_link.obj",
    "leg_link1": "meshes/visual/leg/leg_link1/obj/leg_link1.obj",
    "leg_link2": "meshes_fixed/leg_link2.obj",
    "leg_link3": "meshes/visual/leg/leg_link3/obj/leg_link3.obj",
    "leg_link4": "meshes/visual/leg/leg_link4/obj/leg_link4.obj",
    "leg_link5": "meshes/visual/torso/torso_base_link/obj/torso_base_link.obj",
    "head_link1": "meshes/visual/head/head_link1/obj/head_link1.obj",
    "head_link2": "meshes/visual/head/head_link2/obj/head_link2.obj",
    "left_arm_link1": "meshes/visual/left_arm/left_arm_link1/obj/left_arm_link1.obj",
    "left_arm_link2": "meshes/visual/left_arm/left_arm_link2/obj/left_arm_link2.obj",
    "left_arm_link3": "meshes/visual/left_arm/left_arm_link3/obj/left_arm_link3.obj",
    "left_arm_link4": "meshes/visual/left_arm/left_arm_link4/obj/left_arm_link4.obj",
    "left_arm_link5": "meshes/visual/left_arm/left_arm_link5/obj/left_arm_link5.obj",
    "left_arm_link6": "meshes/visual/left_arm/left_arm_link6/obj/left_arm_link6.obj",
    "left_arm_link7": "meshes/visual/left_arm/left_arm_link7/obj/left_arm_link7.obj",
    "right_arm_link1": "meshes/visual/right_arm/right_arm_link1/obj/right_arm_link1.obj",
    "right_arm_link2": "meshes/visual/right_arm/right_arm_link2/obj/right_arm_link2.obj",
    "right_arm_link3": "meshes/visual/right_arm/right_arm_link3/obj/right_arm_link3.obj",
    "right_arm_link4": "meshes/visual/right_arm/right_arm_link4/obj/right_arm_link4.obj",
    "right_arm_link5": "meshes/visual/right_arm/right_arm_link5/obj/right_arm_link5.obj",
    "right_arm_link6": "meshes/visual/right_arm/right_arm_link6/obj/right_arm_link6.obj",
    "right_arm_link7": "meshes/visual/right_arm/right_arm_link7/obj/right_arm_link7.obj",
}


def ensure_mesh_link(asset_dir: Path, mesh_source: Path) -> None:
    mesh_link = asset_dir / "meshes" / "visual"
    mesh_link.parent.mkdir(parents=True, exist_ok=True)
    if mesh_link.is_symlink():
        if mesh_link.resolve() != mesh_source.resolve():
            mesh_link.unlink()
            mesh_link.symlink_to(mesh_source.resolve())
    elif not mesh_link.exists():
        mesh_link.symlink_to(mesh_source.resolve())


def _clean_obj_mesh(src: Path, dst: Path, *, skip_objects: frozenset[str]) -> None:
    """Drop line-only / degenerate sub-meshes that MuJoCo rejects."""
    filtered: list[str] = []
    skip_current = False
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("o "):
            skip_current = line[2:].strip() in skip_objects
        if skip_current or line.startswith("l "):
            continue
        filtered.append(line)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(filtered) + "\n", encoding="utf-8")


def ensure_fixed_meshes(asset_dir: Path, mesh_source: Path) -> None:
    leg_link2_src = mesh_source / "leg" / "leg_link2" / "obj" / "leg_link2.obj"
    leg_link2_dst = asset_dir / "meshes_fixed" / "leg_link2.obj"
    if not leg_link2_src.is_file():
        raise FileNotFoundError(f"Missing leg_link2 mesh: {leg_link2_src}")
    _clean_obj_mesh(leg_link2_src, leg_link2_dst, skip_objects=LEG_LINK2_SKIP_OBJECTS)


def _extract_robot_tree(raw_xml: str) -> tuple[str, str]:
    start = raw_xml.find('<body name="leg_link1"')
    if start < 0:
        raise RuntimeError("Could not find leg_link1 body in exported MJCF")

    chassis_block = raw_xml[raw_xml.find("<worldbody>") + len("<worldbody>") : start]
    chassis_geoms = "\n".join(
        "    " + line.strip()
        for line in chassis_block.splitlines()
        if line.strip().startswith("<geom")
    )

    depth = 0
    end = start
    for idx in range(start, len(raw_xml)):
        if raw_xml.startswith("<body", idx):
            depth += 1
        elif raw_xml.startswith("</body>", idx):
            depth -= 1
            if depth == 0:
                end = idx + len("</body>")
                break

    return chassis_geoms, raw_xml[start:end].strip()


def _swap_spheres_for_meshes(tree_xml: str) -> str:
    """Replace per-link sphere geoms with one visual mesh geom per body."""
    lines = tree_xml.splitlines()
    out: list[str] = []
    body_stack: list[str] = []
    mesh_added: set[str] = set()

    for line in lines:
        stripped = line.strip()
        body_open = re.match(r'<body name="([^"]+)"', stripped)
        if body_open:
            body_stack.append(body_open.group(1))
            out.append(line)
            continue
        if stripped == "</body>":
            if body_stack:
                current_body = body_stack[-1]
                if current_body in BODY_MESHES and current_body not in mesh_added:
                    indent = line[: len(line) - len(line.lstrip())]
                    out.append(
                        f'{indent}  <geom type="mesh" mesh="{current_body}" class="galbot_visual"/>'
                    )
                    mesh_added.add(current_body)
                body_stack.pop()
            out.append(line)
            continue

        current_body = body_stack[-1] if body_stack else None
        if current_body in BODY_MESHES and re.match(r"<geom\s", stripped):
            if current_body not in mesh_added:
                indent = line[: len(line) - len(line.lstrip())]
                out.append(
                    f'{indent}<geom type="mesh" mesh="{current_body}" class="galbot_visual"/>'
                )
                mesh_added.add(current_body)
            continue

        out.append(line)

    missing = set(BODY_MESHES) - mesh_added - {"base_link"}
    if missing:
        raise RuntimeError(f"Missing mesh geoms for bodies: {sorted(missing)}")

    return "\n".join(out)


def _mesh_assets_xml() -> str:
    lines = []
    for body_name, mesh_file in BODY_MESHES.items():
        lines.append(f'    <mesh name="{body_name}" file="{mesh_file}"/>')
    return "\n".join(lines)


def build_mjcf(urdf_path: Path, asset_dir: Path, mesh_source: Path) -> str:
    ensure_mesh_link(asset_dir, mesh_source)
    ensure_fixed_meshes(asset_dir, mesh_source)

    model = mj.MjModel.from_xml_path(str(urdf_path))
    raw_path = asset_dir / "galbot_one_golf_sphere.mujoco_raw.xml"
    mj.mj_saveLastXML(str(raw_path), model)

    raw_xml = raw_path.read_text(encoding="utf-8")
    _, leg_tree = _extract_robot_tree(raw_xml)
    leg_tree = _swap_spheres_for_meshes(leg_tree)
    leg_tree_indented = "\n".join("      " + line for line in leg_tree.splitlines())

    base_mesh = '      <geom type="mesh" mesh="base_link" class="galbot_visual"/>'

    return f"""<mujoco model="galbot_one_golf">
  <compiler angle="radian" meshdir="."/>

  <default>
    <default class="galbot_visual">
      <geom contype="0" conaffinity="0" rgba="0.82 0.84 0.88 1"/>
    </default>
    <default class="galbot_joint">
      <joint damping="1.0" armature="0.01"/>
      <position kp="800" kv="80"/>
    </default>
    <default class="galbot_base">
      <joint damping="5.0" armature="0.05"/>
      <position kp="2000" kv="200"/>
    </default>
    <default class="galbot_arm">
      <joint damping="0.5" armature="0.001"/>
      <position kp="1500" kv="150"/>
    </default>
    <default class="galbot_leg">
      <joint damping="2.0" armature="0.05"/>
      <position kp="5000" kv="500"/>
    </default>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.5 0.6" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="MatPlane" texture="texplane" texrepeat="4 4" reflectance="0.2"/>
{_mesh_assets_xml()}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="MatPlane" contype="1" conaffinity="1"/>

    <body name="base_link" pos="0 0 0" childclass="galbot_visual">
      <joint name="base_x" type="slide" axis="1 0 0" class="galbot_base"/>
      <joint name="base_y" type="slide" axis="0 1 0" class="galbot_base"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" class="galbot_base"/>
      <inertial pos="0.02644 -0.00449 0.07524" mass="8.04" diaginertia="0.331314 0.295737 0.195089"/>
{base_mesh}
{leg_tree_indented}
    </body>
  </worldbody>

  <actuator>
    <position name="actuator_base_x" joint="base_x" class="galbot_base"/>
    <position name="actuator_base_y" joint="base_y" class="galbot_base"/>
    <position name="actuator_base_yaw" joint="base_yaw" class="galbot_base"/>
    <position name="actuator_leg_joint1" joint="leg_joint1" class="galbot_leg"/>
    <position name="actuator_leg_joint2" joint="leg_joint2" class="galbot_leg"/>
    <position name="actuator_leg_joint3" joint="leg_joint3" class="galbot_leg"/>
    <position name="actuator_leg_joint4" joint="leg_joint4" class="galbot_leg"/>
    <position name="actuator_leg_joint5" joint="leg_joint5" class="galbot_leg"/>
    <position name="actuator_head_joint1" joint="head_joint1" class="galbot_joint"/>
    <position name="actuator_head_joint2" joint="head_joint2" class="galbot_joint"/>
    <position name="actuator_left_arm_joint1" joint="left_arm_joint1" class="galbot_arm"/>
    <position name="actuator_left_arm_joint2" joint="left_arm_joint2" class="galbot_arm"/>
    <position name="actuator_left_arm_joint3" joint="left_arm_joint3" class="galbot_arm"/>
    <position name="actuator_left_arm_joint4" joint="left_arm_joint4" class="galbot_arm"/>
    <position name="actuator_left_arm_joint5" joint="left_arm_joint5" class="galbot_arm"/>
    <position name="actuator_left_arm_joint6" joint="left_arm_joint6" class="galbot_arm"/>
    <position name="actuator_left_arm_joint7" joint="left_arm_joint7" class="galbot_arm"/>
    <position name="actuator_right_arm_joint1" joint="right_arm_joint1" class="galbot_arm"/>
    <position name="actuator_right_arm_joint2" joint="right_arm_joint2" class="galbot_arm"/>
    <position name="actuator_right_arm_joint3" joint="right_arm_joint3" class="galbot_arm"/>
    <position name="actuator_right_arm_joint4" joint="right_arm_joint4" class="galbot_arm"/>
    <position name="actuator_right_arm_joint5" joint="right_arm_joint5" class="galbot_arm"/>
    <position name="actuator_right_arm_joint6" joint="right_arm_joint6" class="galbot_arm"/>
    <position name="actuator_right_arm_joint7" joint="right_arm_joint7" class="galbot_arm"/>
  </actuator>
</mujoco>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-source", type=Path, default=DEFAULT_MESH_SRC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    xml_text = build_mjcf(args.urdf, args.output.parent, args.mesh_source)
    args.output.write_text(xml_text, encoding="utf-8")

    model = mj.MjModel.from_xml_path(str(args.output))
    mesh_geoms = sum(1 for i in range(model.ngeom) if model.geom_type[i] == mj.mjtGeom.mjGEOM_MESH)
    sphere_geoms = sum(1 for i in range(model.ngeom) if model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE)
    print(f"Wrote {args.output}")
    print(f"  nv={model.nv}, nbody={model.nbody}, nu={model.nu}, ngeom={model.ngeom}")
    print(f"  mesh geoms={mesh_geoms}, sphere geoms={sphere_geoms}")


if __name__ == "__main__":
    main()

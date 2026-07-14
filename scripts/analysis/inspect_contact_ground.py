"""Print merged contact_ground config and body/geom resolution for a robot."""

from __future__ import annotations

import argparse
import json

import mujoco as mj
from rich import print

from general_motion_retargeting.contact_ground import ContactGroundPipeline
from general_motion_retargeting.contact_ground_config import (
    build_contact_ground_config,
    robot_preset,
)
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect contact_ground robot config.")
    parser.add_argument("--robot", required=True, type=str)
    parser.add_argument(
        "--src_human",
        default="bvh_lafan1",
        help="Human format key in IK_CONFIG_DICT, e.g. bvh_lafan1 or smplx.",
    )
    parser.add_argument(
        "--enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override enabled flag when showing merged config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = args.robot
    if args.src_human not in IK_CONFIG_DICT:
        raise SystemExit(f"Unknown src_human: {args.src_human}")
    if robot not in IK_CONFIG_DICT[args.src_human]:
        raise SystemExit(f"No IK config for {args.src_human} -> {robot}")

    with open(IK_CONFIG_DICT[args.src_human][robot], encoding="utf-8") as f:
        ik_config = json.load(f)

    merged = build_contact_ground_config(ik_config, robot, cli_override=args.enabled)
    preset = robot_preset(robot)
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    pipeline = ContactGroundPipeline(merged, model, fps=30.0)

    print(f"[bold]Robot[/bold]: {robot}")
    print(f"[bold]IK config[/bold]: {IK_CONFIG_DICT[args.src_human][robot]}")
    print("[bold]Preset only[/bold]:")
    print(json.dumps(preset, indent=2))
    print("[bold]Merged config[/bold]:")
    print(json.dumps(merged, indent=2))
    print(
        f"[bold]Resolved geoms[/bold]: foot={len(pipeline.foot_geom_ids)}, "
        f"trunk={len(pipeline.trunk_geom_ids)}, leg={len(pipeline.leg_geom_ids)}, "
        f"ground={len(pipeline.ground_geom_ids)}, lying={len(pipeline.lying_ground_geom_ids)}"
    )
    if pipeline.missing_bodies:
        print("[red]Missing bodies[/red]:", pipeline.missing_bodies)
    else:
        print("[green]All configured bodies and floor geom resolved.[/green]")


if __name__ == "__main__":
    main()

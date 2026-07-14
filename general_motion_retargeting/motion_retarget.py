
import mink
import mujoco as mj
import numpy as np
import json
from typing import Optional
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT, PLANAR_BASE_ROBOTS
from .contact_ground import ContactGroundPipeline, FootGroundLimit
from .contact_ground_config import build_contact_ground_config
from rich import print

G1_SELF_COLLISION_PAIRS = [
    # left arm vs torso / pelvis
    (
        ["left_elbow_yaw_collision", "left_wrist_collision", "left_hand_collision"],
        ["torso_collision", "pelvis_collision"],
    ),

    # right arm vs torso / pelvis
    (
        ["right_elbow_yaw_collision", "right_wrist_collision", "right_hand_collision"],
        ["torso_collision", "pelvis_collision"],
    ),

    # left arm vs right arm
    (
        ["left_elbow_yaw_collision", "left_wrist_collision", "left_hand_collision"],
        ["right_elbow_yaw_collision", "right_wrist_collision", "right_hand_collision"],
    ),

    # left hand / wrist vs left leg
    (
        ["left_wrist_collision", "left_hand_collision"],
        ["left_thigh_collision", "left_shin_collision"],
    ),

    # right hand / wrist vs right leg
    (
        ["right_wrist_collision", "right_hand_collision"],
        ["right_thigh_collision", "right_shin_collision"],
    ),

    # left hand / wrist vs right leg, for cross-body motion
    (
        ["left_wrist_collision", "left_hand_collision"],
        ["right_thigh_collision", "right_shin_collision"],
    ),

    # right hand / wrist vs left leg, for cross-body motion
    (
        ["right_wrist_collision", "right_hand_collision"],
        ["left_thigh_collision", "left_shin_collision"],
    ),

    # left leg vs right leg
    (
        ["left_thigh_collision", "left_shin_collision"],
        ["right_thigh_collision", "right_shin_collision"],
    ),

    # left foot vs right foot
    (
        [
            "left_foot1_collision",
            "left_foot2_collision",
            "left_foot3_collision",
            "left_foot4_collision",
            "left_foot5_collision",
            "left_foot6_collision",
            "left_foot7_collision",
        ],
        [
            "right_foot1_collision",
            "right_foot2_collision",
            "right_foot3_collision",
            "right_foot4_collision",
            "right_foot5_collision",
            "right_foot6_collision",
            "right_foot7_collision",
        ],
    ),

    # hand / arm vs head
    (
        ["left_elbow_yaw_collision", "left_wrist_collision", "left_hand_collision"],
        ["head_collision"],
    ),
    (
        ["right_elbow_yaw_collision", "right_wrist_collision", "right_hand_collision"],
        ["head_collision"],
    ),
]

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
        contact_ground: Optional[bool] = None,
        foot_ground_limit: Optional[bool] = None,
        fix_robot_penetration: Optional[bool] = None,
        motion_fps: float = 30.0,
    ) -> None:
        self.verbose = verbose

        # load the robot model
        self.tgt_robot = tgt_robot
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        self.actual_human_height = actual_human_height
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.planar_base_cfg = ik_config.get("planar_base")
        self._g1_reference = None

        self.max_iter = 15

        self.solver = solver
        self.damping = damping

        self.task_frames1 = []
        self.task_frames2 = []
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        collision_cfg = ik_config.get("collision_avoidance", {})
        collision_enabled = bool(collision_cfg.get("enabled", False))
        collision_pairs = collision_cfg.get("self_collision_pairs", [])

        # unitree_g1: use collision pairs from IK JSON when present; else built-in list (legacy configs).
        if tgt_robot == "unitree_g1" and not collision_pairs:
            collision_pairs = G1_SELF_COLLISION_PAIRS
            collision_enabled = True

        if collision_enabled:
            valid_collision_pairs = self._filter_valid_collision_pairs(collision_pairs)
            if valid_collision_pairs:
                collision_avoidance_limit = mink.CollisionAvoidanceLimit(
                    model=self.model,
                    geom_pairs=valid_collision_pairs,  # type: ignore
                    minimum_distance_from_collisions=collision_cfg.get("min_distance", 0.005),
                    collision_detection_distance=collision_cfg.get("detection_distance", 0.15),
                    gain=collision_cfg.get("gain", 0.85),
                    bound_relaxation=collision_cfg.get("bound_relaxation", 0.0),
                )

                print("[GMR] collision pairs",collision_pairs)
                self.ik_limits.append(collision_avoidance_limit)
            elif self.verbose:
                print("[GMR] Collision avoidance enabled but no valid geom pairs were found. Skip this limit.")

        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

        contact_ground_cfg = build_contact_ground_config(
            ik_config,
            tgt_robot,
            cli_override=contact_ground,
        )
        if fix_robot_penetration is not None:
            contact_ground_cfg["fix_robot_penetration"] = bool(fix_robot_penetration)
        self.contact_ground = ContactGroundPipeline(
            contact_ground_cfg,
            self.model,
            fps=motion_fps,
        )
        foot_ground_cfg = {
            "enabled": False,
            "ground_z": contact_ground_cfg.get("ground_z", 0.0),
            "margin": 0.01,
            "gain": 0.95,
        }
        foot_ground_cfg.update(dict(ik_config.get("foot_ground_limit", {})))
        if foot_ground_limit is not None:
            foot_ground_cfg["enabled"] = bool(foot_ground_limit)

        foot_ground_limit_obj = None
        if bool(foot_ground_cfg.get("enabled", False)):
            foot_ground_limit_obj = FootGroundLimit(
                self.model,
                self.contact_ground.foot_geom_ids,
                ground_z=float(foot_ground_cfg.get("ground_z", 0.0)),
                margin=float(foot_ground_cfg.get("margin", 0.01)),
                gain=float(foot_ground_cfg.get("gain", 0.95)),
            )
            if foot_ground_limit_obj.geom_ids:
                self.ik_limits.append(foot_ground_limit_obj)

        if self.verbose and self.contact_ground.enabled:
            print(
                "[GMR] contact_ground enabled: "
                f"robot={tgt_robot}, "
                f"feet={self.contact_ground.foot_bodies}, "
                f"foot_geoms={len(self.contact_ground.foot_geom_ids)}, "
                f"trunk_geoms={len(self.contact_ground.trunk_geom_ids)}, "
                f"leg_geoms={len(self.contact_ground.leg_geom_ids)}, "
                f"foot_bodies={len(self.contact_ground.foot_body_ids)}"
            )
            if self.contact_ground.missing_bodies:
                print(
                    "[GMR] contact_ground warning: unresolved bodies "
                    f"{self.contact_ground.missing_bodies}"
                )
        if self.verbose and foot_ground_limit_obj is not None:
            print(
                "[GMR] foot_ground_limit enabled: "
                f"geoms={len(foot_ground_limit_obj.geom_ids)}, "
                f"margin={foot_ground_limit_obj.margin}, "
                f"gain={foot_ground_limit_obj.gain}"
            )

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        self.task_frames1 = []
        self.task_frames2 = []
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.task_frames1.append(
                    {
                        "task": task,
                        "human_body": body_name,
                        "robot_frame": frame_name,
                        "pos_offset": np.array(pos_offset) - self.ground,
                        "rot_offset": R.from_quat(
                            self._quat_wxyz_to_xyzw(rot_offset)
                        ),
                    }
                )
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    self._quat_wxyz_to_xyzw(rot_offset)
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.task_frames2.append(
                    {
                        "task": task,
                        "human_body": body_name,
                        "robot_frame": frame_name,
                        "pos_offset": np.array(pos_offset) - self.ground,
                        "rot_offset": R.from_quat(
                            self._quat_wxyz_to_xyzw(rot_offset)
                        ),
                    }
                )
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    self._quat_wxyz_to_xyzw(rot_offset)
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

    def _apply_body_offset(self, pos, quat_wxyz, pos_offset, rot_offset):
        pos = np.asarray(pos, dtype=float)
        updated_rot = R.from_quat(self._quat_wxyz_to_xyzw(quat_wxyz)) * rot_offset
        updated_quat = self._quat_xyzw_to_wxyz(updated_rot.as_quat())
        global_pos_offset = updated_rot.apply(np.asarray(pos_offset, dtype=float))
        return pos + global_pos_offset, updated_quat

    def _body_target_from_entry(self, human_data, entry):
        pos, quat = human_data[entry["human_body"]]
        return self._apply_body_offset(
            pos, quat, entry["pos_offset"], entry["rot_offset"]
        )

    def _planar_base_target(self, pos, quat_wxyz):
        """Project a human root target onto the ground plane with yaw-only orientation."""
        cfg = self.planar_base_cfg or {}
        ground_z = float(cfg.get("ground_z", self.ground[2]))
        pos = np.asarray(pos, dtype=float)
        rot = R.from_quat(self._quat_wxyz_to_xyzw(quat_wxyz))
        if cfg.get("yaw_frame") == "g1_pelvis":
            # Match unitree_g1 smplx pelvis heading (same rot_offset as smplx_to_g1.json).
            rot = rot * R.from_quat([-0.5, -0.5, -0.5, 0.5])
        yaw = rot.as_euler("ZYX")[0]
        rot_yaw = R.from_euler("Z", yaw)
        return np.array([pos[0], pos[1], ground_z], dtype=float), self._quat_xyzw_to_wxyz(
            rot_yaw.as_quat()
        )

    def _resolve_ik_target(self, entry, human_data):
        pos, rot = self._body_target_from_entry(human_data, entry)
        if (
            self.tgt_robot in PLANAR_BASE_ROBOTS
            and self.planar_base_cfg is not None
            and entry["robot_frame"] == self.planar_base_cfg.get("frame_name", self.robot_root_name)
            and entry["human_body"] == self.planar_base_cfg.get("human_body", self.human_root_name)
        ):
            pos, rot = self._planar_base_target(pos, rot)
        return pos, rot

    def _build_scaled_human_data(self, human_data):
        scaled = {
            body_name: [np.asarray(pos), np.asarray(quat)]
            for body_name, (pos, quat) in human_data.items()
        }
        planar_frame = (
            self.planar_base_cfg.get("frame_name")
            if self.planar_base_cfg is not None
            else None
        )
        for entry in self.task_frames1:
            if entry["robot_frame"] == planar_frame:
                continue
            pos, rot = self._body_target_from_entry(human_data, entry)
            scaled[entry["human_body"]] = [pos, rot]
        for entry in self.task_frames2:
            pos, rot = self._body_target_from_entry(human_data, entry)
            scaled[entry["human_body"]] = [pos, rot]
        return scaled

    @staticmethod
    def _quat_wxyz_to_xyzw(quat):
        quat = np.asarray(quat)
        return quat[[1, 2, 3, 0]]

    @staticmethod
    def _quat_xyzw_to_wxyz(quat):
        quat = np.asarray(quat)
        return quat[[3, 0, 1, 2]]

    def _filter_valid_collision_pairs(self, collision_pairs):
        geom_name_set = set()
        for i in range(self.model.ngeom):
            geom_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
            if geom_name is not None:
                geom_name_set.add(geom_name)

        def normalize_geom_group(group):
            if isinstance(group, (list, tuple)):
                return list(group)
            return [group]

        def is_valid_geom_ref(geom_ref):
            if isinstance(geom_ref, int):
                return 0 <= geom_ref < self.model.ngeom
            if isinstance(geom_ref, str):
                return geom_ref in geom_name_set
            return False

        valid_pairs = []
        for pair in collision_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                if self.verbose:
                    print(f"[GMR] Skip invalid collision pair format: {pair}")
                continue

            left_group = normalize_geom_group(pair[0])
            right_group = normalize_geom_group(pair[1])

            missing = [g for g in left_group + right_group if not is_valid_geom_ref(g)]
            if missing:
                if self.verbose:
                    print(f"[GMR] Skip collision pair with missing geoms: {missing}")
                continue

            valid_pairs.append((left_group, right_group))

        return valid_pairs

  
    def _prepare_scaled_human_data(self, human_data, offset_to_ground=False):
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.apply_ground_offset(human_data)
        if self.contact_ground.enabled:
            human_data = self.contact_ground.process_human_frame(human_data)
        else:
            self.contact_ground.observe_human_frame(human_data)
        if not self.contact_ground.enabled and offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        return human_data

    def _get_g1_root_xy(self, human_data):
        if self._g1_reference is None:
            self._g1_reference = GeneralMotionRetargeting(
                actual_human_height=self.actual_human_height,
                src_human="smplx",
                tgt_robot="unitree_g1",
                verbose=False,
            )
        q = self._g1_reference.retarget(human_data)
        return float(q[0]), float(q[1])

    def _snap_planar_base_qpos(self, human_data, raw_human_data=None):
        if self.tgt_robot not in PLANAR_BASE_ROBOTS or not self.planar_base_cfg:
            return None
        frame_name = self.planar_base_cfg.get("frame_name", self.robot_root_name)
        entry = next((e for e in self.task_frames1 if e["robot_frame"] == frame_name), None)
        if entry is None:
            return None
        pos, rot = self._resolve_ik_target(entry, human_data)
        if self.planar_base_cfg.get("position_source") == "g1_root":
            g1_input = raw_human_data if raw_human_data is not None else human_data
            pos[0], pos[1] = self._get_g1_root_xy(g1_input)
        yaw = R.from_quat(self._quat_wxyz_to_xyzw(rot)).as_euler("ZYX")[0]
        qpos = self.configuration.data.qpos
        qpos[0] = pos[0]
        qpos[1] = pos[1]
        qpos[2] = yaw
        mj.mj_forward(self.model, self.configuration.data)
        return entry["task"]

    def _run_ik_tasks(self, tasks, max_iter=None, freeze_base=False, base_qpos=None):
        if not tasks:
            return
        max_iter = self.max_iter if max_iter is None else max_iter
        curr_error = np.linalg.norm(
            np.concatenate([task.compute_error(self.configuration) for task in tasks])
        )
        dt = self.configuration.model.opt.timestep
        num_iter = 0
        while num_iter < max_iter:
            vel = mink.solve_ik(
                self.configuration, tasks, dt, self.solver, self.damping, limits=self.ik_limits
            )
            if freeze_base and base_qpos is not None:
                vel[:3] = 0.0
            self.configuration.integrate_inplace(vel, dt)
            if freeze_base and base_qpos is not None:
                self.configuration.data.qpos[:3] = base_qpos
                mj.mj_forward(self.model, self.configuration.data)
            next_error = np.linalg.norm(
                np.concatenate([task.compute_error(self.configuration) for task in tasks])
            )
            if curr_error - next_error <= 0.001:
                break
            curr_error = next_error
            num_iter += 1

    def update_targets(self, human_data, offset_to_ground=False):
        human_data = self._prepare_scaled_human_data(human_data, offset_to_ground)
        self.scaled_human_data = self._build_scaled_human_data(human_data)

        if self.use_ik_match_table1:
            for entry in self.task_frames1:
                pos, rot = self._resolve_ik_target(entry, human_data)
                entry["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )
        
        if self.use_ik_match_table2:
            for entry in self.task_frames2:
                pos, rot = self._resolve_ik_target(entry, human_data)
                entry["task"].set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        freeze_base = self.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.planar_base_cfg)
        base_qpos = None
        if freeze_base:
            scaled_human = self._prepare_scaled_human_data(human_data, offset_to_ground)
            self._snap_planar_base_qpos(scaled_human, raw_human_data=human_data)
            base_qpos = self.configuration.data.qpos[:3].copy()

        if self.use_ik_match_table1:
            self._run_ik_tasks(
                self.tasks1,
                freeze_base=freeze_base,
                base_qpos=base_qpos,
            )

        if self.use_ik_match_table2:
            self._run_ik_tasks(
                self.tasks2,
                freeze_base=freeze_base,
                base_qpos=base_qpos,
            )

        if self.contact_ground.fix_penetration:
            self.contact_ground.fix_robot_penetration(self.model, self.configuration.data)

        if freeze_base and base_qpos is not None:
            self.configuration.data.qpos[:3] = base_qpos
            mj.mj_forward(self.model, self.configuration.data)

        return self.configuration.data.qpos.copy()

    def set_motion_fps(self, fps: float) -> None:
        if self.contact_ground.enabled:
            self.contact_ground.set_fps(fps)


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if body_name not in rot_offsets:
                offset_human_data[body_name] = [np.asarray(pos), np.asarray(quat)]
                continue
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_rot = R.from_quat(self._quat_wxyz_to_xyzw(quat)) * rot_offsets[body_name]
            updated_quat = self._quat_xyzw_to_wxyz(updated_rot.as_quat())
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(self._quat_wxyz_to_xyzw(updated_quat)).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name] = [pos - np.array([0, 0, self.ground_offset]), quat]
        return human_data

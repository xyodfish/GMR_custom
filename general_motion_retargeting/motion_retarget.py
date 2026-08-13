
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
        control_feasibility: bool = False,
        cf_margin: float = 0.2,
        cf_mode: str = "torque",
        cf_uniform_accel_cap: float = 30.0,
    ) -> None:
        self.verbose = verbose

        # load the robot model
        self.tgt_robot = tgt_robot
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        if verbose:
            print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")

        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        if verbose:
            print("[GMR] Robot Body names and their IDs:")

        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        if verbose:
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
        self.mobile_upper_body_cfg = ik_config.get("mobile_upper_body")
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
        self._setup_mobile_upper_body()
        
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

        self.contact_ground.foot_ground_limit_enabled = foot_ground_limit_obj is not None

        # Control-feasibility limiter: keep committed joint torques away from actuator
        # saturation (upgrades the objective from kinematic smoothness to control feasibility).
        self.control_feasibility = bool(control_feasibility)
        self._cf_q_prev = None
        self.cf_limiter = None
        if self.control_feasibility:
            from .control_feasibility import TorqueFeasibilityLimiter
            self.cf_limiter = TorqueFeasibilityLimiter(
                self.model,
                margin=cf_margin,
                mode=cf_mode,
                uniform_accel_cap=cf_uniform_accel_cap,
                fps=motion_fps,
            )
            if self.verbose:
                print(
                    f"[GMR] control_feasibility enabled: mode={cf_mode}, "
                    f"margin={cf_margin}, joints={self.cf_limiter.qadr.size}"
                )

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

    def _setup_mobile_upper_body(self):
        self.mobile_upper_body_tasks = None
        if self.mobile_upper_body_cfg is None:
            return

        if self.tgt_robot not in PLANAR_BASE_ROBOTS or self.planar_base_cfg is None:
            raise ValueError("mobile_upper_body requires a planar-base robot configuration")

        cfg = self.mobile_upper_body_cfg
        torso_frame = cfg["torso_frame"]
        head_frame = cfg["head_frame"]
        arm_chains = cfg["arm_chains"]
        required_frames = {torso_frame, head_frame}
        for chain in arm_chains:
            required_frames.update(
                {
                    chain["shoulder_frame"],
                    chain["elbow_frame"],
                    chain["wrist_frame"],
                    chain["orientation_frame"],
                }
            )

        missing_frames = sorted(required_frames - self.robot_body_names.keys())
        if missing_frames:
            raise ValueError(f"mobile_upper_body references missing robot frames: {missing_frames}")

        torso_task = mink.FrameTask(
            frame_name=torso_frame,
            frame_type="body",
            position_cost=float(cfg.get("torso_position_cost", 120.0)),
            orientation_cost=float(cfg.get("torso_orientation_cost", 30.0)),
            lm_damping=1,
        )
        head_task = mink.FrameTask(
            frame_name=head_frame,
            frame_type="body",
            position_cost=0.0,
            orientation_cost=float(cfg.get("head_orientation_cost", 3.0)),
            lm_damping=1,
        )

        neutral_data = mj.MjData(self.model)
        neutral_data.qpos[:] = self.model.qpos0
        mj.mj_forward(self.model, neutral_data)
        arm_tasks = []
        for chain in arm_chains:
            shoulder_id = self.robot_body_names[chain["shoulder_frame"]]
            elbow_id = self.robot_body_names[chain["elbow_frame"]]
            wrist_id = self.robot_body_names[chain["wrist_frame"]]
            upper_length = np.linalg.norm(
                neutral_data.xpos[elbow_id] - neutral_data.xpos[shoulder_id]
            )
            forearm_length = np.linalg.norm(
                neutral_data.xpos[wrist_id] - neutral_data.xpos[elbow_id]
            )
            if upper_length <= 0.0 or forearm_length <= 0.0:
                raise ValueError(
                    f"mobile_upper_body arm chain has zero-length segment: {chain}"
                )

            arm_tasks.append(
                {
                    **chain,
                    "upper_length": upper_length,
                    "forearm_length": forearm_length,
                    "elbow_task": mink.FrameTask(
                        frame_name=chain["elbow_frame"],
                        frame_type="body",
                        position_cost=float(cfg.get("arm_position_cost", 120.0)),
                        orientation_cost=float(cfg.get("elbow_orientation_cost", 1.0)),
                        lm_damping=1,
                    ),
                    "wrist_task": mink.FrameTask(
                        frame_name=chain["wrist_frame"],
                        frame_type="body",
                        position_cost=float(cfg.get("arm_position_cost", 120.0)),
                        orientation_cost=0.0,
                        lm_damping=1,
                    ),
                    "orientation_task": mink.FrameTask(
                        frame_name=chain["orientation_frame"],
                        frame_type="body",
                        position_cost=0.0,
                        orientation_cost=float(cfg.get("wrist_orientation_cost", 2.0)),
                        lm_damping=1,
                    ),
                }
            )

        posture_cost = np.full(self.model.nv, float(cfg.get("posture_cost", 0.05)))
        posture_cost[:3] = 0.0
        for joint_name, cost in cfg.get("joint_posture_cost", {}).items():
            if joint_name not in self.robot_dof_names:
                raise ValueError(
                    f"mobile_upper_body posture references missing joint: {joint_name}"
                )

            posture_cost[self.robot_dof_names[joint_name]] = float(cost)

        posture_task = mink.PostureTask(self.model, cost=posture_cost, lm_damping=1)
        posture_task.set_target(self.model.qpos0)
        torso_id = self.robot_body_names[torso_frame]
        torso_neutral_rotation = R.from_matrix(
            neutral_data.xmat[torso_id].reshape(3, 3)
        )
        head_id = self.robot_body_names[head_frame]
        head_neutral_rotation = R.from_matrix(
            neutral_data.xmat[head_id].reshape(3, 3)
        )
        self.mobile_upper_body_tasks = {
            "torso": torso_task,
            "head": head_task,
            "arms": arm_tasks,
            "posture": posture_task,
            "torso_neutral_rotation": torso_neutral_rotation,
            "head_neutral_relative_rotation": (
                torso_neutral_rotation.inv() * head_neutral_rotation
            ),
        }
        self._mobile_initialized = False

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
        yaw = self._rotation_yaw(rot)
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
        if self.use_ik_match_table2:
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

    @staticmethod
    def _rotation_yaw(rotation):
        x, y, z, w = rotation.as_quat()
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

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
        yaw = self._rotation_yaw(R.from_quat(self._quat_wxyz_to_xyzw(rot)))
        qpos = self.configuration.data.qpos
        qpos[0] = pos[0]
        qpos[1] = pos[1]
        qpos[2] = yaw
        mj.mj_forward(self.model, self.configuration.data)
        return entry["task"]

    @staticmethod
    def _normalized_direction(start, end, label):
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        norm = np.linalg.norm(direction)
        if norm <= 1e-8:
            raise ValueError(f"Cannot retarget zero-length human segment: {label}")

        return direction / norm

    def _set_frame_task_target(self, task, pos, rotation):
        quat_wxyz = self._quat_xyzw_to_wxyz(rotation.as_quat())
        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(quat_wxyz), np.asarray(pos, dtype=float)
            )
        )

    def _mobile_body_rotation(self, human_data, body_name, offset_wxyz):
        rotation = R.from_quat(self._quat_wxyz_to_xyzw(human_data[body_name][1]))
        offset = R.from_quat(self._quat_wxyz_to_xyzw(offset_wxyz))
        return rotation * offset

    def _set_mobile_torso_targets(self, raw_human_data, base_qpos):
        cfg = self.mobile_upper_body_cfg
        tasks = self.mobile_upper_body_tasks
        required_bodies = {
            cfg["torso_human_body"],
            cfg["head_human_body"],
        }
        for chain in tasks["arms"]:
            required_bodies.update(
                {
                    chain["shoulder_human_body"],
                    chain["elbow_human_body"],
                    chain["wrist_human_body"],
                }
            )

        missing_bodies = sorted(required_bodies - raw_human_data.keys())
        if missing_bodies:
            raise ValueError(
                f"mobile_upper_body input is missing human bodies: {missing_bodies}"
            )

        base_rotation = R.from_euler("Z", base_qpos[2])
        torso_rotation = self._mobile_body_rotation(
            raw_human_data,
            cfg["torso_human_body"],
            cfg.get("torso_rotation_offset", [1.0, 0.0, 0.0, 0.0]),
        )
        neutral_rotation = tasks["torso_neutral_rotation"]
        relative_rotation = base_rotation.inv() * torso_rotation * neutral_rotation.inv()
        relative_euler = relative_rotation.as_euler("XYZ")
        orientation_limit = np.deg2rad(
            np.asarray(cfg.get("torso_orientation_limit_deg", [20.0, 20.0, 30.0]))
        )
        relative_euler = np.clip(relative_euler, -orientation_limit, orientation_limit)
        torso_rotation = (
            base_rotation * R.from_euler("XYZ", relative_euler) * neutral_rotation
        )

        source_height = float(raw_human_data[cfg["torso_human_body"]][0][2])
        torso_height = source_height * float(cfg.get("torso_height_scale", 0.75))
        min_height, max_height = cfg.get("torso_height_range", [0.72, 1.12])
        torso_height = np.clip(torso_height, float(min_height), float(max_height))
        local_xy = np.asarray(cfg.get("torso_local_xy", [0.107, 0.0]), dtype=float)
        torso_pos = np.array([base_qpos[0], base_qpos[1], 0.0])
        torso_pos[:2] += base_rotation.apply([local_xy[0], local_xy[1], 0.0])[:2]
        torso_pos[2] = torso_height
        self._set_frame_task_target(tasks["torso"], torso_pos, torso_rotation)

        human_torso_rotation = R.from_quat(
            self._quat_wxyz_to_xyzw(raw_human_data[cfg["torso_human_body"]][1])
        )
        human_head_rotation = R.from_quat(
            self._quat_wxyz_to_xyzw(raw_human_data[cfg["head_human_body"]][1])
        )
        head_relative_euler = (
            human_torso_rotation.inv() * human_head_rotation
        ).as_euler("XYZ")
        head_limit = np.deg2rad(
            np.asarray(cfg.get("head_orientation_limit_deg", [30.0, 30.0, 60.0]))
        )
        head_relative_euler = np.clip(
            head_relative_euler, -head_limit, head_limit
        )
        head_rotation = (
            torso_rotation
            * R.from_euler("XYZ", head_relative_euler)
            * tasks["head_neutral_relative_rotation"]
        )
        self._set_frame_task_target(tasks["head"], np.zeros(3), head_rotation)

    def _set_mobile_arm_targets(self, raw_human_data):
        tasks = self.mobile_upper_body_tasks
        for chain in tasks["arms"]:
            shoulder_body = chain["shoulder_human_body"]
            elbow_body = chain["elbow_human_body"]
            wrist_body = chain["wrist_human_body"]
            upper_direction = self._normalized_direction(
                raw_human_data[shoulder_body][0],
                raw_human_data[elbow_body][0],
                f"{shoulder_body}->{elbow_body}",
            )
            forearm_direction = self._normalized_direction(
                raw_human_data[elbow_body][0],
                raw_human_data[wrist_body][0],
                f"{elbow_body}->{wrist_body}",
            )
            shoulder_id = self.robot_body_names[chain["shoulder_frame"]]
            shoulder_pos = self.configuration.data.xpos[shoulder_id].copy()
            elbow_pos = shoulder_pos + chain["upper_length"] * upper_direction
            wrist_pos = elbow_pos + chain["forearm_length"] * forearm_direction
            elbow_rotation = self._mobile_body_rotation(
                raw_human_data, elbow_body, chain["elbow_rotation_offset"]
            )
            wrist_rotation = self._mobile_body_rotation(
                raw_human_data, wrist_body, chain["wrist_rotation_offset"]
            )
            self._set_frame_task_target(
                chain["elbow_task"], elbow_pos, elbow_rotation
            )
            self._set_frame_task_target(
                chain["wrist_task"], wrist_pos, R.identity()
            )
            self._set_frame_task_target(
                chain["orientation_task"], np.zeros(3), wrist_rotation
            )

    def _run_mobile_upper_body(self, raw_human_data, base_qpos):
        tasks = self.mobile_upper_body_tasks
        self._set_mobile_torso_targets(raw_human_data, base_qpos)
        torso_tasks = [tasks["torso"], tasks["head"], tasks["posture"]]
        if self._mobile_initialized:
            torso_max_iterations = int(
                self.mobile_upper_body_cfg.get("torso_iterations", 20)
            )
            torso_min_iterations = int(
                self.mobile_upper_body_cfg.get("torso_min_iterations", 4)
            )
            arm_max_iterations = int(
                self.mobile_upper_body_cfg.get("arm_iterations", 15)
            )
            arm_min_iterations = int(
                self.mobile_upper_body_cfg.get("arm_min_iterations", 3)
            )
        else:
            torso_max_iterations = int(
                self.mobile_upper_body_cfg.get("initial_torso_iterations", 60)
            )
            torso_min_iterations = int(
                self.mobile_upper_body_cfg.get("initial_torso_min_iterations", 30)
            )
            arm_max_iterations = int(
                self.mobile_upper_body_cfg.get("initial_arm_iterations", 40)
            )
            arm_min_iterations = int(
                self.mobile_upper_body_cfg.get("initial_arm_min_iterations", 20)
            )

        self._run_ik_tasks(
            torso_tasks,
            max_iter=torso_max_iterations,
            min_iter=torso_min_iterations,
            freeze_base=True,
            base_qpos=base_qpos,
        )

        all_tasks = [tasks["torso"], tasks["head"]]
        for chain in tasks["arms"]:
            all_tasks.extend(
                [chain["elbow_task"], chain["wrist_task"], chain["orientation_task"]]
            )

        all_tasks.append(tasks["posture"])
        for _ in range(int(self.mobile_upper_body_cfg.get("arm_target_passes", 2))):
            self._set_mobile_arm_targets(raw_human_data)
            self._run_ik_tasks(
                all_tasks,
                max_iter=arm_max_iterations,
                min_iter=arm_min_iterations,
                freeze_base=True,
                base_qpos=base_qpos,
            )

        self._mobile_initialized = True

    def _apply_mobile_joint_margin(self):
        if self.mobile_upper_body_cfg is None:
            return

        margin = np.deg2rad(
            float(self.mobile_upper_body_cfg.get("joint_limit_margin_deg", 0.0))
        )
        if margin <= 0.0:
            return

        qpos = self.configuration.data.qpos
        for joint_id in range(self.model.njnt):
            if not self.model.jnt_limited[joint_id]:
                continue

            if self.model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_HINGE:
                continue

            lower, upper = self.model.jnt_range[joint_id]
            if upper - lower <= 2.0 * margin:
                continue

            qpos_id = self.model.jnt_qposadr[joint_id]
            qpos[qpos_id] = np.clip(qpos[qpos_id], lower + margin, upper - margin)

        mj.mj_forward(self.model, self.configuration.data)

    def _run_ik_tasks(
        self, tasks, max_iter=None, min_iter=0, freeze_base=False, base_qpos=None
    ):
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
            num_iter += 1
            if num_iter >= min_iter and curr_error - next_error <= 0.001:
                break
            curr_error = next_error

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
        return human_data

            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        scaled_human = self.update_targets(human_data, offset_to_ground)

        freeze_base = self.tgt_robot in PLANAR_BASE_ROBOTS and bool(self.planar_base_cfg)
        base_qpos = None
        if freeze_base:
            self._snap_planar_base_qpos(scaled_human, raw_human_data=human_data)
            base_qpos = self.configuration.data.qpos[:3].copy()

        if self.mobile_upper_body_tasks is not None:
            self._run_mobile_upper_body(human_data, base_qpos)
            self._apply_mobile_joint_margin()
        elif self.use_ik_match_table1:
            self._run_ik_tasks(
                self.tasks1,
                freeze_base=freeze_base,
                base_qpos=base_qpos,
            )

        if self.mobile_upper_body_tasks is None and self.use_ik_match_table2:
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

        if self.cf_limiter is not None:
            q_raw = self.configuration.data.qpos.copy()
            q_prev = self._cf_q_prev if self._cf_q_prev is not None else q_raw
            q_feasible = self.cf_limiter.project(self.configuration.data, q_raw, q_prev)
            self.configuration.data.qpos[:] = q_feasible
            mj.mj_forward(self.model, self.configuration.data)
            self._cf_q_prev = q_feasible.copy()

        return self.configuration.data.qpos.copy()

    def set_motion_fps(self, fps: float) -> None:
        if self.contact_ground.enabled:
            self.contact_ground.set_fps(fps)
        if self.cf_limiter is not None:
            self.cf_limiter.set_fps(fps)


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

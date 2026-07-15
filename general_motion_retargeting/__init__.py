from types import SimpleNamespace

import numpy as np
from rich import print

if not hasattr(np, "dtypes"):
    np.dtypes = SimpleNamespace()

if not hasattr(np, "exceptions"):
    np.exceptions = SimpleNamespace(ComplexWarning=np.ComplexWarning)

from .params import IK_CONFIG_ROOT, ASSET_ROOT, ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT, PLANAR_BASE_ROBOTS
from .motion_retarget import GeneralMotionRetargeting
from .sliding_window_retarget import SlidingWindowConfig, SlidingWindowRetargeter
from .trajectory_optimization_retarget import (
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationRetargeter,
)
from .robot_motion_viewer import RobotMotionViewer, draw_frame
from .data_loader import load_robot_motion
from .kinematics_model import KinematicsModel

from .neck_retarget import human_head_to_robot_neck

try:
    from .xrobot_utils import XRobotStreamer, XRobotRecorder
except ImportError:
    print("XRobotStreamer is not installed. Please install xrobotoolkit_sdk to use this feature.")
    XRobotStreamer = None
    XRobotRecorder = None

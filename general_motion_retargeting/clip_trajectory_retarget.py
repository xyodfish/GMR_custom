"""Deprecated: use ``batch_trajectory_retarget`` instead."""

import warnings

warnings.warn(
    "general_motion_retargeting.clip_trajectory_retarget is deprecated; "
    "use batch_trajectory_retarget",
    DeprecationWarning,
    stacklevel=2,
)

from .batch_trajectory_retarget import *  # noqa: F401,F403

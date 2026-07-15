#!/usr/bin/env python3
"""Deprecated wrapper — use ``to_robot_batch.py`` instead."""

import pathlib
import runpy
import sys
import warnings

warnings.warn(
    "to_robot_clip.py is deprecated; use to_robot_batch.py",
    DeprecationWarning,
    stacklevel=1,
)

runpy.run_path(
    str(pathlib.Path(__file__).resolve().parent / "to_robot_batch.py"),
    run_name="__main__",
)

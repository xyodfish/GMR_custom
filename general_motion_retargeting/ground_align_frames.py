"""Offline ground alignment for human motion frames (GVHMR / SMPL-X).

Fixes common monocular float / root-height drift by shifting every body
translation along Z. Two modes:

- ``lower_envelope`` (default): pull the local lower-foot envelope to ground.
  Best for GVHMR global-height drift; compresses true aerial phases somewhat.
- ``support_hold``: update offset only on support frames (speed + near-envelope),
  hold/interpolate while airborne. Gentler on real jumps/runs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

FootMode = Literal["lower_envelope", "support_hold"]

_DEFAULT_FOOT_NAMES = ("left_foot", "right_foot", "left_ankle", "right_ankle")


def _body_pos(frame: dict, name: str) -> np.ndarray | None:
    if name not in frame:
        return None
    return np.asarray(frame[name][0], dtype=np.float64).reshape(3)


def _resolve_foot_names(frames: list[dict], foot_names: tuple[str, ...] | None) -> list[str]:
    names = list(foot_names or _DEFAULT_FOOT_NAMES)
    if not frames:
        return names
    present = [n for n in names if n in frames[0]]
    if present:
        return present
    # Last resort: any key containing foot/ankle
    found = [
        k
        for k in frames[0]
        if ("foot" in k.lower() or "ankle" in k.lower()) and "hand" not in k.lower()
    ]
    if not found:
        raise ValueError("No foot/ankle bodies found for ground alignment")
    return found


def _foot_z_series(frames: list[dict], foot_names: list[str]) -> np.ndarray:
    """Per-frame min foot height, shape (T,)."""
    t = len(frames)
    z = np.full(t, np.inf, dtype=np.float64)
    for i, fr in enumerate(frames):
        vals = []
        for name in foot_names:
            p = _body_pos(fr, name)
            if p is not None:
                vals.append(float(p[2]))
        if vals:
            z[i] = min(vals)
    if not np.isfinite(z).any():
        raise ValueError("Could not read any foot Z values")
    # Replace missing with neighbor
    bad = ~np.isfinite(z)
    if bad.any():
        good = np.where(~bad)[0]
        z[bad] = np.interp(np.where(bad)[0], good, z[good])
    return z


def _rolling_min(x: np.ndarray, half_win: int) -> np.ndarray:
    if half_win <= 0:
        return x.copy()
    t = len(x)
    out = np.empty(t, dtype=np.float64)
    for i in range(t):
        lo = max(0, i - half_win)
        hi = min(t, i + half_win + 1)
        out[i] = float(np.min(x[lo:hi]))
    return out


def _lpf(x: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 1e-6, 1.0))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1.0 - a) * y[i - 1]
    return y


def _apply_offset(frames: list[dict], offsets: np.ndarray) -> list[dict]:
    out: list[dict] = []
    for fr, dz in zip(frames, offsets):
        shifted: dict = {}
        for name, (pos, quat) in fr.items():
            p = np.asarray(pos, dtype=np.float64).reshape(3).copy()
            p[2] -= float(dz)
            shifted[name] = (p, quat)
        out.append(shifted)
    return out


def compute_ground_align_offsets(
    frames: list[dict],
    fps: float,
    *,
    mode: FootMode = "lower_envelope",
    ground_z: float = 0.0,
    ground_margin: float = 0.03,
    foot_names: tuple[str, ...] | None = None,
    window_s: float = 0.40,
    lpf_alpha: float = 0.35,
    support_band: float = 0.06,
    vel_threshold: float = 0.55,
) -> tuple[np.ndarray, dict]:
    """Return per-frame Z offsets (subtract from body translations) and debug stats."""
    if not frames:
        return np.zeros(0), {}

    feet = _resolve_foot_names(frames, foot_names)
    z_min = _foot_z_series(frames, feet)
    gz = float(ground_z)
    target = gz + float(ground_margin)
    half = max(1, int(round(float(window_s) * float(fps) * 0.5)))
    env = _rolling_min(z_min, half)

    if mode == "lower_envelope":
        # Always park the local lower envelope on the ground.
        raw = env - target
        offsets = _lpf(raw, lpf_alpha)
    elif mode == "support_hold":
        # Contact if near local envelope and not moving too fast vertically.
        dt = 1.0 / max(float(fps), 1e-6)
        vz = np.zeros_like(z_min)
        vz[1:] = (z_min[1:] - z_min[:-1]) / dt
        near_env = (z_min - env) <= float(support_band)
        slow = np.abs(vz) <= float(vel_threshold)
        contact = near_env & slow

        raw = np.zeros_like(z_min)
        # Seed with first contact or first frame
        first = int(np.argmax(contact)) if contact.any() else 0
        raw[0] = z_min[first] - target
        last = raw[0]
        for i in range(len(z_min)):
            if contact[i]:
                last = z_min[i] - target
            raw[i] = last
        # Fill gaps by interpolating between contact anchors (reduces mid-air drift)
        idx = np.where(contact)[0]
        if len(idx) >= 2:
            anchor_off = z_min[idx] - target
            raw = np.interp(np.arange(len(z_min)), idx, anchor_off)
            # Outside anchors: hold edge
            raw[: idx[0]] = anchor_off[0]
            raw[idx[-1] :] = anchor_off[-1]
        offsets = _lpf(raw, lpf_alpha)
    else:
        raise ValueError(f"Unknown ground_align mode: {mode}")

    # Hard floor: never push any foot keypoint below ground_z (LPF can overshoot).
    offsets = np.minimum(offsets, z_min - gz)

    z_after = z_min - offsets
    stats = {
        "mode": mode,
        "foot_names": feet,
        "z_min_before_mean": float(z_min.mean()),
        "z_min_before_min": float(z_min.min()),
        "z_min_after_mean": float(z_after.mean()),
        "z_min_after_min": float(z_after.min()),
        "float_pct_before": float((z_min > 0.05).mean() * 100.0),
        "float_pct_after": float((z_after > 0.05).mean() * 100.0),
        "pen_pct_after": float((z_after < gz).mean() * 100.0),
        "offset_mean": float(offsets.mean()),
        "offset_max": float(np.max(np.abs(offsets))),
    }
    return offsets, stats


def ground_align_human_frames(
    frames: list[dict],
    fps: float,
    *,
    mode: FootMode = "lower_envelope",
    ground_z: float = 0.0,
    ground_margin: float = 0.03,
    foot_names: tuple[str, ...] | None = None,
    window_s: float = 0.40,
    lpf_alpha: float = 0.35,
    support_band: float = 0.06,
    vel_threshold: float = 0.55,
    verbose: bool = False,
) -> list[dict]:
    """Return a new frame list with Z-aligned translations."""
    offsets, stats = compute_ground_align_offsets(
        frames,
        fps,
        mode=mode,
        ground_z=ground_z,
        ground_margin=ground_margin,
        foot_names=foot_names,
        window_s=window_s,
        lpf_alpha=lpf_alpha,
        support_band=support_band,
        vel_threshold=vel_threshold,
    )
    if verbose:
        print(
            f"[ground_align] mode={stats['mode']} "
            f"float%>5cm {stats['float_pct_before']:.0f}→{stats['float_pct_after']:.0f} "
            f"z_min_mean {stats['z_min_before_mean']:.3f}→{stats['z_min_after_mean']:.3f} "
            f"offset_mean={stats['offset_mean']:.3f} "
            f"z_min_after_min={stats['z_min_after_min']:.3f}"
        )
    return _apply_offset(frames, offsets)

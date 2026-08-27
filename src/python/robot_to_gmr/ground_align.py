"""Ground-align HumanFrame-style dicts (position/orientation objects)."""

from __future__ import annotations

import numpy as np


def ground_align_frames(
    frames: list[dict[str, dict[str, list[float]]]],
    *,
    foot_names: tuple[str, ...] = ("left_foot", "right_foot"),
    ground_z: float = 0.0,
    mode: str = "lower_envelope",
) -> list[dict[str, dict[str, list[float]]]]:
    """Shift all body positions along Z so the foot envelope sits on ``ground_z``.

    ``lower_envelope``: one global offset = min foot z over the clip (preserves aerial relative height).
    """
    if not frames:
        return frames

    foot_z: list[float] = []
    for frame in frames:
        vals = []
        for name in foot_names:
            if name in frame:
                vals.append(float(frame[name]["position"][2]))

        if vals:
            foot_z.append(min(vals))

    if not foot_z:
        return frames

    if mode == "lower_envelope":
        offset = float(np.min(foot_z)) - float(ground_z)
    else:
        raise ValueError(f"Unsupported ground align mode: {mode}")

    if abs(offset) < 1e-6:
        return frames

    out: list[dict[str, dict[str, list[float]]]] = []
    for frame in frames:
        shifted: dict[str, dict[str, list[float]]] = {}
        for name, pose in frame.items():
            pos = list(pose["position"])
            pos[2] = float(pos[2]) - offset
            shifted[name] = {
                "position": pos,
                "orientation": list(pose["orientation"]),
            }

        out.append(shifted)

    return out


def _foot_collision_geom_ids(model, foot_geom_substr: str = "foot") -> list[int]:
    """Collect contype>0 foot geoms (name/body contains foot/toe/ankle sole tokens).

    Accepts both ``*_foot*_collision`` (Unitree) and ``cylinder_foot_*`` / unnamed
    foot cylinders on foot bodies (Fourier GR3), plus ``toe*_left`` style geoms.
    """
    import mujoco

    tokens = (foot_geom_substr, "toe", "sole", "ankle")
    geom_ids: list[int] = []
    for i in range(model.ngeom):
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue

        if int(model.geom_contype[i]) == 0:
            continue

        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        body = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        ).lower()
        blob = f"{name} {body}"
        if not any(tok in blob for tok in tokens):
            continue

        geom_ids.append(i)

    if not geom_ids:
        raise ValueError("No foot collision geoms found")

    return geom_ids


def _geom_side_label(model, geom_id: int) -> str | None:
    """Return 'left' / 'right' from geom or parent body name, else None."""
    import mujoco

    name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
    body = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
    ).lower()
    return _side_from_text(f"{name} {body}")


def _side_from_text(text: str) -> str | None:
    b = text.lower()
    # Order matters: check explicit left/right before short l/r tokens.
    if "left" in b or "leg_l" in b:
        return "left"

    if "right" in b or "leg_r" in b:
        return "right"

    # l_ankle / r_ankle / ankle_*_l_link / leg_l6
    if (
        b.startswith("l_")
        or "_l_" in b
        or b.endswith("_l")
        or b.endswith("_l_link")
        or "ankle_l" in b
        or "foot_l" in b
        or "toe_l" in b
    ):
        return "left"

    if (
        b.startswith("r_")
        or "_r_" in b
        or b.endswith("_r")
        or b.endswith("_r_link")
        or "ankle_r" in b
        or "foot_r" in b
        or "toe_r" in b
    ):
        return "right"

    return None


def _sole_z(model, data, geom_ids: list[int]) -> float:
    """Lowest world-Z point of the given geoms (capsule/sphere/box/mesh aware)."""
    import mujoco

    zs: list[float] = []
    for i in geom_ids:
        pos = np.asarray(data.geom_xpos[i], dtype=np.float64)
        mat = np.asarray(data.geom_xmat[i].reshape(3, 3), dtype=np.float64)
        typ = int(model.geom_type[i])
        size = model.geom_size[i]
        if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            zs.append(float(pos[2] - size[0]))
        elif typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            # MuJoCo capsule: radius=size[0], half-length=size[1] along local +Z.
            axis = mat[:, 2]
            half = float(size[1])
            radius = float(size[0])
            zs.append(float(min(pos[2] + axis[2] * half, pos[2] - axis[2] * half) - radius))
        elif typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            # Cylinder: flat caps; subtract radius projected into world Z.
            axis = mat[:, 2]
            half = float(size[1])
            radius = float(size[0])
            radial = radius * float(np.sqrt(max(0.0, 1.0 - axis[2] * axis[2])))
            zs.append(float(min(pos[2] + axis[2] * half, pos[2] - axis[2] * half) - radial))
        elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
            # Conservative: center z minus projected half-extents.
            half = np.abs(mat) @ np.asarray(size[:3], dtype=np.float64)
            zs.append(float(pos[2] - half[2]))
        elif typ == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[i])
            vadr = int(model.mesh_vertadr[mid])
            vnum = int(model.mesh_vertnum[mid])
            verts = model.mesh_vert[vadr : vadr + vnum]
            world = verts @ mat.T + pos
            zs.append(float(np.min(world[:, 2])))
        else:
            zs.append(float(pos[2] - size[0]))

    if not zs:
        raise ValueError("No sole samples")

    return float(min(zs))


def _mesh_sole_z_for_body(model, data, body_id: int) -> float:
    """Lowest visual-mesh Z on a body (fallback when no foot collision geoms)."""
    import mujoco

    zs: list[float] = []
    for i in range(model.ngeom):
        if int(model.geom_bodyid[i]) != int(body_id):
            continue

        if int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue

        mid = int(model.geom_dataid[i])
        vadr = int(model.mesh_vertadr[mid])
        vnum = int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[vadr : vadr + vnum]
        mat = data.geom_xmat[i].reshape(3, 3)
        pos = data.geom_xpos[i]
        world = verts @ mat.T + pos
        zs.append(float(np.min(world[:, 2])))

    if not zs:
        return float(data.xpos[body_id][2])

    return float(min(zs))


def _resolve_ankle_hinge_ids(model, side: str) -> list[int]:
    """Hinge joints that articulate the foot for ``side`` in ('left','right')."""
    import mujoco

    tokens = (
        f"{side}_ankle",
        f"{side}_foot",
    )
    out: list[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if any(tok in name for tok in tokens):
            out.append(j)

    return out


def flatten_stance_feet_ik(
    qpos_frames: np.ndarray,
    model_xml: str,
    contacts: list[dict[str, bool]],
    *,
    iterations: int = 10,
    step: float = 0.7,
    ground_z: float = 0.0,
) -> np.ndarray:
    """For contacting feet: ankle IK so sole is level (+Z up) and near ``ground_z``.

    Fixes robots whose foot IK EE is proximal to ankle pitch (pitch stuck at 0), and
    reduces one-foot float after a root-only ground snap.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos [T,{model.nq}], got {qpos.shape}")

    body_ids = _resolve_foot_body_ids(model)
    side_of = {"left_foot": "left", "right_foot": "right"}
    joint_ids = {name: _resolve_ankle_hinge_ids(model, side_of[name]) for name in body_ids}
    if not any(joint_ids.values()):
        return qpos

    # Prefer foot collision geoms when present; else mesh sole on foot body.
    try:
        all_foot_geoms = _foot_collision_geom_ids(model)
    except ValueError:
        all_foot_geoms = []

    geoms_by_side = {
        "left_foot": [i for i in all_foot_geoms if _geom_side_label(model, i) == "left"],
        "right_foot": [i for i in all_foot_geoms if _geom_side_label(model, i) == "right"],
    }

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    up = np.array([0.0, 0.0, 1.0])

    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        for _ in range(iterations):
            mujoco.mj_forward(model, data)
            moved = False
            for name, body_id in body_ids.items():
                if not contacts[t].get(name, False):
                    continue

                j_ids = joint_ids[name]
                if not j_ids:
                    continue

                # Orientation: foot body +Z toward world up.
                R = data.xmat[body_id].reshape(3, 3)
                foot_z = np.asarray(R[:, 2], dtype=np.float64)
                ori_err = np.cross(foot_z, up)

                # Height: sole to ground.
                if geoms_by_side[name]:
                    sole = _sole_z(model, data, geoms_by_side[name])
                else:
                    sole = _mesh_sole_z_for_body(model, data, body_id)

                height_err = float(ground_z - sole)
                if float(np.linalg.norm(ori_err)) < 1e-3 and abs(height_err) < 1e-4:
                    continue

                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                cols = [int(model.jnt_dofadr[j]) for j in j_ids]
                j_ori = jacr[:, cols]
                j_z = jacp[2:3, cols]
                # Height dominates: floating stance feet are the visual bug; ori is soft.
                A = np.vstack([10.0 * j_z, 0.25 * j_ori])
                b = np.array(
                    [10.0 * height_err, 0.25 * ori_err[0], 0.25 * ori_err[1], 0.25 * ori_err[2]],
                    dtype=np.float64,
                )
                dq, *_ = np.linalg.lstsq(A, b, rcond=None)
                for j_id, delta in zip(j_ids, dq):
                    qadr = int(model.jnt_qposadr[j_id])
                    data.qpos[qadr] = float(data.qpos[qadr] + step * delta)
                    if model.jnt_limited[j_id]:
                        lo, hi = model.jnt_range[j_id]
                        data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))

                moved = True

            if not moved:
                break

        qpos[t] = data.qpos

    return qpos


def _resolve_foot_body_ids(model) -> dict[str, int]:
    import mujoco

    candidates = {
        "left_foot": (
            # Prefer bodies that typically carry sole collision / visual foot.
            "left_ankle_pitch_link",
            "left_foot_roll_link",
            "left_ankle_roll_link",
            "left_foot_pitch_link",
            "left_sole_link",
            "left_foot_link",
            "left_foot",
            "LeftFoot",
            "left_toe_link",
            "toeLeft",
            "leg_left_ankle_roll",
            "leg_left_ankle_pitch",
            "l_ankle_roll_link",
            "l_ankle_pitch_link",
            "ankle_roll_l_link",
            "ankle_pitch_l_link",
            "anklePitchLeft",
            "leg_l6_link",
            "left_ankle_link",
        ),
        "right_foot": (
            "right_ankle_pitch_link",
            "right_foot_roll_link",
            "right_ankle_roll_link",
            "right_foot_pitch_link",
            "right_sole_link",
            "right_foot_link",
            "right_foot",
            "RightFoot",
            "right_toe_link",
            "toeRight",
            "leg_right_ankle_roll",
            "leg_right_ankle_pitch",
            "r_ankle_roll_link",
            "r_ankle_pitch_link",
            "ankle_roll_r_link",
            "ankle_pitch_r_link",
            "anklePitchRight",
            "leg_r6_link",
            "right_ankle_link",
        ),
    }
    out: dict[str, int] = {}
    for key, names in candidates.items():
        for name in names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                out[key] = int(body_id)
                break

        if key not in out:
            want = "left" if key == "left_foot" else "right"
            for body_id in range(model.nbody):
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                side = _side_from_text(bname)
                if side != want:
                    continue

                low = bname.lower()
                if any(tok in low for tok in ("foot", "ankle", "sole", "toe")):
                    out[key] = int(body_id)
                    break

        if key not in out:
            raise ValueError(f"Could not resolve body for {key}")

    return out


def snap_robot_qpos_to_ground(
    qpos_frames: np.ndarray,
    model_xml: str,
    *,
    foot_geom_substr: str = "foot",
    mode: str = "per_frame",
    contacts: list[dict[str, bool]] | None = None,
) -> np.ndarray:
    """Lower free-root Z so foot collision soles meet z=0.

    ``per_frame``: each frame's lowest sole is placed on the ground (best for walk viz).
    ``global_min``: one offset from the clip-wide minimum sole height.
    If ``contacts`` is provided with ``per_frame``, only contacting feet are used when available.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos [T,{model.nq}], got {qpos.shape}")

    try:
        geom_ids = _foot_collision_geom_ids(model, foot_geom_substr)
        left_geoms = [i for i in geom_ids if _geom_side_label(model, i) == "left"]
        right_geoms = [i for i in geom_ids if _geom_side_label(model, i) == "right"]
        use_mesh = False
    except ValueError:
        body_ids = _resolve_foot_body_ids(model)
        use_mesh = True
        left_geoms = []
        right_geoms = []
        geom_ids = []

    sole_z = np.zeros(qpos.shape[0], dtype=np.float64)
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        if use_mesh:
            use_bodies = list(body_ids.values())
            if contacts is not None and t < len(contacts) and mode == "per_frame":
                use_bodies = []
                if contacts[t].get("left_foot", False):
                    use_bodies.append(body_ids["left_foot"])

                if contacts[t].get("right_foot", False):
                    use_bodies.append(body_ids["right_foot"])

                if not use_bodies:
                    use_bodies = list(body_ids.values())

            sole_z[t] = min(_mesh_sole_z_for_body(model, data, bid) for bid in use_bodies)
        else:
            use_ids = geom_ids
            if contacts is not None and t < len(contacts) and mode == "per_frame":
                ids: list[int] = []
                if contacts[t].get("left_foot", False):
                    ids.extend(left_geoms)

                if contacts[t].get("right_foot", False):
                    ids.extend(right_geoms)

                if ids:
                    use_ids = ids

            sole_z[t] = _sole_z(model, data, use_ids)

    if mode == "global_min":
        qpos[:, 2] -= float(np.min(sole_z))
    elif mode == "per_frame":
        qpos[:, 2] -= sole_z
    else:
        raise ValueError(f"Unsupported snap mode: {mode}")

    return qpos


def copy_joints_by_name(
    qpos_dst: np.ndarray,
    model_dst_xml: str,
    qpos_src: np.ndarray,
    model_src_xml: str,
    *,
    name_tokens: tuple[str, ...] = ("ankle", "foot_pitch", "foot_roll"),
) -> np.ndarray:
    """Copy matching hinge joints from src→dst when joint names share tokens.

    Used to transfer G1 ankle pitch/roll onto robots whose foot IK EE was proximal
    (pitch stuck) or whose foot frame offsets differ from G1.
    """
    import mujoco

    md = mujoco.MjModel.from_xml_path(model_dst_xml)
    ms = mujoco.MjModel.from_xml_path(model_src_xml)
    qpos = np.asarray(qpos_dst, dtype=np.float64).copy()
    src = np.asarray(qpos_src, dtype=np.float64)
    n = min(len(qpos), len(src))

    src_by_name: dict[str, int] = {}
    for j in range(ms.njnt):
        if ms.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        name = mujoco.mj_id2name(ms, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        src_by_name[name] = int(ms.jnt_qposadr[j])

    for j in range(md.njnt):
        if md.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        name = mujoco.mj_id2name(md, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name not in src_by_name:
            continue

        if not any(tok in name for tok in name_tokens):
            continue

        qadr_d = int(md.jnt_qposadr[j])
        qadr_s = src_by_name[name]
        vals = src[:n, qadr_s].copy()
        if md.jnt_limited[j]:
            lo, hi = md.jnt_range[j]
            vals = np.clip(vals, lo, hi)

        qpos[:n, qadr_d] = vals

    return qpos


def level_contact_soles_ik(
    qpos_frames: np.ndarray,
    model_xml: str,
    contacts: list[dict[str, bool]],
    *,
    iterations: int = 8,
    step: float = 0.85,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Lower each contacting foot's sole to ``ground_z`` via ankle pitch/roll only.

    Intended after G1 ankle copy + root snap: removes residual one-foot float when
    both feet are marked in contact.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    body_ids = _resolve_foot_body_ids(model)
    side_of = {"left_foot": "left", "right_foot": "right"}
    joint_ids = {name: _resolve_ankle_hinge_ids(model, side_of[name]) for name in body_ids}
    try:
        all_foot_geoms = _foot_collision_geom_ids(model)
    except ValueError:
        all_foot_geoms = []

    geoms_by_side = {
        "left_foot": [i for i in all_foot_geoms if _geom_side_label(model, i) == "left"],
        "right_foot": [i for i in all_foot_geoms if _geom_side_label(model, i) == "right"],
    }
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        for _ in range(iterations):
            mujoco.mj_forward(model, data)
            moved = False
            for name, body_id in body_ids.items():
                if not contacts[t].get(name, False):
                    continue

                j_ids = joint_ids[name]
                if not j_ids:
                    continue

                if geoms_by_side[name]:
                    sole = _sole_z(model, data, geoms_by_side[name])
                else:
                    sole = _mesh_sole_z_for_body(model, data, body_id)

                height_err = float(ground_z - sole)
                if abs(height_err) < 5e-4:
                    continue

                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                cols = [int(model.jnt_dofadr[j]) for j in j_ids]
                j_z = jacp[2:3, cols]
                dq, *_ = np.linalg.lstsq(j_z, np.array([height_err]), rcond=None)
                for j_id, delta in zip(j_ids, dq):
                    qadr = int(model.jnt_qposadr[j_id])
                    data.qpos[qadr] = float(data.qpos[qadr] + step * delta)
                    if model.jnt_limited[j_id]:
                        lo, hi = model.jnt_range[j_id]
                        data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))

                moved = True

            if not moved:
                break

        qpos[t] = data.qpos

    return qpos


def infer_foot_contacts_from_soles(
    qpos_frames: np.ndarray,
    model_xml: str,
    *,
    height_tol: float = 0.02,
) -> list[dict[str, bool]]:
    """Mark a foot in contact when its sole is within ``height_tol`` of the lower sole."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64)
    body_ids = _resolve_foot_body_ids(model)
    try:
        all_foot_geoms = _foot_collision_geom_ids(model)
    except ValueError:
        all_foot_geoms = []

    geoms_by_side = {
        "left_foot": [i for i in all_foot_geoms if _geom_side_label(model, i) == "left"],
        "right_foot": [i for i in all_foot_geoms if _geom_side_label(model, i) == "right"],
    }

    contacts: list[dict[str, bool]] = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        soles: dict[str, float] = {}
        for name, body_id in body_ids.items():
            if geoms_by_side[name]:
                soles[name] = _sole_z(model, data, geoms_by_side[name])
            else:
                soles[name] = _mesh_sole_z_for_body(model, data, body_id)

        z_min = min(soles.values())
        contacts.append({name: (soles[name] - z_min) <= float(height_tol) for name in soles})

    return contacts


def retarget_root_xy_from_reference(
    qpos_frames: np.ndarray,
    reference_xy: np.ndarray,
) -> np.ndarray:
    """Overwrite free-root XY with a reference trajectory (canonical pelvis / source root)."""
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    ref = np.asarray(reference_xy, dtype=np.float64)
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(f"reference_xy must be [T,2], got {ref.shape}")

    n = min(qpos.shape[0], ref.shape[0])
    qpos[:n, 0:2] = ref[:n]
    return qpos


def smooth_joint_qpos(qpos_frames: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average hinge joints (qpos[7:]) to reduce IK jitter; keep free root."""
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    if window <= 1 or qpos.shape[0] < 3:
        return qpos

    half = window // 2
    joints = qpos[:, 7:].copy()
    smoothed = joints.copy()
    for t in range(qpos.shape[0]):
        t0 = max(0, t - half)
        t1 = min(qpos.shape[0], t + half + 1)
        smoothed[t] = joints[t0:t1].mean(axis=0)

    qpos[:, 7:] = smoothed
    return qpos


def smooth_joint_qpos_model(
    qpos_frames: np.ndarray,
    model_xml: str,
    window: int = 5,
    *,
    skip_name_tokens: tuple[str, ...] = ("wrist_pitch", "wrist_yaw"),
) -> np.ndarray:
    """Smooth hinges like ``smooth_joint_qpos``, but leave named joints unsmoothed."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    raw = np.asarray(qpos_frames, dtype=np.float64)
    qpos = smooth_joint_qpos(raw, window=window)
    if window <= 1 or raw.shape[0] < 3 or not skip_name_tokens:
        return qpos

    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not any(token in name for token in skip_name_tokens):
            continue

        qadr = int(model.jnt_qposadr[j])
        qpos[:, qadr] = raw[:, qadr]

    return qpos


def align_wrists_to_forearm(
    qpos_frames: np.ndarray,
    model_xml: str,
    *,
    iterations: int = 12,
    step: float = 0.8,
) -> np.ndarray:
    """Keep wrist roll; solve pitch/yaw so wrist_yaw +X aligns with elbow→wrist_roll.

    Matching G1/SMPL-X wrist world orientation on H2 leaves a visible kink because H2's
    forearm axis (elbow→roll) is not colinear with the wrist frame +X the way G1's is.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos [T,{model.nq}], got {qpos.shape}")

    sides = ("left", "right")
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    joint_ids: dict[str, dict[str, int]] = {}
    body_ids: dict[str, dict[str, int]] = {}
    for side in sides:
        joint_ids[side] = {
            "pitch": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_pitch_joint"),
            "yaw": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_yaw_joint"),
        }
        body_ids[side] = {
            "elbow": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link"),
            "roll": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link"),
            "yaw": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"),
        }
        if any(v < 0 for v in joint_ids[side].values()) or any(v < 0 for v in body_ids[side].values()):
            raise ValueError(f"Missing wrist bodies/joints for side={side} in {model_xml}")

    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        for _ in range(iterations):
            mujoco.mj_forward(model, data)
            moved = False
            for side in sides:
                elbow = np.asarray(data.xpos[body_ids[side]["elbow"]], dtype=np.float64)
                roll = np.asarray(data.xpos[body_ids[side]["roll"]], dtype=np.float64)
                forearm = roll - elbow
                n = float(np.linalg.norm(forearm))
                if n < 1e-8:
                    continue

                forearm /= n
                yaw_body = body_ids[side]["yaw"]
                hand_x = np.asarray(data.xmat[yaw_body].reshape(3, 3)[:, 0], dtype=np.float64)
                # Rotate hand_x toward forearm; error axis ~ hand_x × forearm.
                err_axis = np.cross(hand_x, forearm)
                err_norm = float(np.linalg.norm(err_axis))
                if err_norm < 1e-4:
                    continue

                mujoco.mj_jacBody(model, data, jacp, jacr, yaw_body)
                pitch_j = joint_ids[side]["pitch"]
                yaw_j = joint_ids[side]["yaw"]
                cols = [int(model.jnt_dofadr[pitch_j]), int(model.jnt_dofadr[yaw_j])]
                # Map joint rates to angular velocity at the yaw link, then to hand_x tip motion.
                j_w = jacr[:, cols]
                # d(hand_x) ≈ omega × hand_x = -[hand_x]_x omega
                hx = np.array(
                    [
                        [0.0, -hand_x[2], hand_x[1]],
                        [hand_x[2], 0.0, -hand_x[0]],
                        [-hand_x[1], hand_x[0], 0.0],
                    ],
                    dtype=np.float64,
                )
                j_dir = (-hx @ j_w)  # 3x2
                target = forearm - hand_x
                dq, *_ = np.linalg.lstsq(j_dir, target, rcond=None)
                for j_id, delta in zip((pitch_j, yaw_j), dq):
                    qadr = int(model.jnt_qposadr[j_id])
                    data.qpos[qadr] = float(data.qpos[qadr] + step * delta)
                    if model.jnt_limited[j_id]:
                        lo, hi = model.jnt_range[j_id]
                        data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))

                moved = True

            if not moved:
                break

        qpos[t] = data.qpos

    return qpos


def lock_stance_feet_xy(
    qpos_frames: np.ndarray,
    model_xml: str,
    contacts: list[dict[str, bool]],
) -> np.ndarray:
    """Shift free-root XY each stance phase so contacting feet do not slip.

    Single stance: fully plant that foot.
    Double support: plant the lower foot (primary), avoid averaging which leaves residual slip.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos [T,{model.nq}], got {qpos.shape}")

    if len(contacts) < qpos.shape[0]:
        raise ValueError("contacts length must be >= number of qpos frames")

    body_ids = _resolve_foot_body_ids(model)
    hold: dict[str, np.ndarray | None] = {"left_foot": None, "right_foot": None}

    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)

        active: list[str] = []
        for name, body_id in body_ids.items():
            in_contact = bool(contacts[t].get(name, False))
            foot_xy = np.asarray(data.xpos[body_id][:2], dtype=np.float64)
            foot_z = float(data.xpos[body_id][2])
            if not in_contact:
                hold[name] = None
                continue

            if hold[name] is None:
                hold[name] = foot_xy.copy()

            active.append(name)

        if not active:
            continue

        if len(active) == 1:
            name = active[0]
            body_id = body_ids[name]
            foot_xy = np.asarray(data.xpos[body_id][:2], dtype=np.float64)
            corr = hold[name] - foot_xy
        else:
            # Double support: plant the currently lower foot.
            zs = {name: float(data.xpos[body_ids[name]][2]) for name in active}
            name = min(zs, key=zs.get)
            body_id = body_ids[name]
            foot_xy = np.asarray(data.xpos[body_id][:2], dtype=np.float64)
            corr = hold[name] - foot_xy

        qpos[t, 0] += float(corr[0])
        qpos[t, 1] += float(corr[1])

    return qpos


def plant_stance_feet_ik(
    qpos_frames: np.ndarray,
    model_xml: str,
    contacts: list[dict[str, bool]],
    *,
    iterations: int = 8,
    step: float = 0.6,
) -> np.ndarray:
    """Reduce stance-foot world XY slip by small hinge-joint IK corrections."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64).copy()
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"Expected qpos [T,{model.nq}], got {qpos.shape}")

    body_ids = _resolve_foot_body_ids(model)
    hold: dict[str, np.ndarray | None] = {name: None for name in body_ids}
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # Only allow lower-body hinges to move for stance-foot planting.
    # This prevents waist/arm compensation while reducing foot slip.
    allowed_joint_tokens = (
        "hip",
        "knee",
        "ankle",
    )
    allowed_v_idx: list[int] = []
    allowed_q_idx: list[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        joint_name = model.joint(j).name or ""
        if not any(token in joint_name for token in allowed_joint_tokens):
            continue

        allowed_v_idx.append(int(model.jnt_dofadr[j]))
        allowed_q_idx.append(int(model.jnt_qposadr[j]))

    if not allowed_v_idx:
        return qpos

    for t in range(qpos.shape[0]):
        for name, body_id in body_ids.items():
            if not contacts[t].get(name, False):
                hold[name] = None

        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)

        for name, body_id in body_ids.items():
            if not contacts[t].get(name, False):
                continue

            foot_xy = np.array(data.xpos[body_id][:2], dtype=np.float64, copy=True)
            if hold[name] is None:
                hold[name] = foot_xy

        for _ in range(iterations):
            mujoco.mj_forward(model, data)
            delta_v = np.zeros(model.nv, dtype=np.float64)
            active = 0
            for name, body_id in body_ids.items():
                if hold[name] is None or not contacts[t].get(name, False):
                    continue

                mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                err = hold[name] - np.array(data.xpos[body_id][:2], dtype=np.float64, copy=True)
                if float(np.linalg.norm(err)) < 1e-4:
                    continue

                # Map XY error through translational Jacobian of allowed lower-body hinges.
                j = jacp[:2, allowed_v_idx]
                dq_allowed, *_ = np.linalg.lstsq(j, err, rcond=None)
                for i, v_idx in enumerate(allowed_v_idx):
                    delta_v[v_idx] += dq_allowed[i]
                active += 1

            if active == 0 or float(np.linalg.norm(delta_v)) < 1e-6:
                break

            # Integrate only selected lower-body hinges.
            for q_idx, v_idx in zip(allowed_q_idx, allowed_v_idx):
                data.qpos[q_idx] = data.qpos[q_idx] + step * delta_v[v_idx]

            # Keep all hinge joints within limits when available.
            for j in range(model.njnt):
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue

                qadr = int(model.jnt_qposadr[j])
                if model.jnt_limited[j]:
                    lo, hi = model.jnt_range[j]
                    data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))

        qpos[t] = data.qpos

    return qpos


def measure_stance_foot_slip_mps(
    qpos_frames: np.ndarray,
    model_xml: str,
    contacts: list[dict[str, bool]],
    fps: float,
) -> float:
    """Mean horizontal speed of contacting feet (m/s)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)
    qpos = np.asarray(qpos_frames, dtype=np.float64)
    body_ids = _resolve_foot_body_ids(model)
    dt = 1.0 / float(fps)
    slips: list[float] = []
    prev_xy: dict[str, np.ndarray | None] = {name: None for name in body_ids}

    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for name, body_id in body_ids.items():
            xy = np.array(data.xpos[body_id][:2], dtype=np.float64, copy=True)
            if (
                t > 0
                and contacts[t].get(name, False)
                and contacts[t - 1].get(name, False)
                and prev_xy[name] is not None
            ):
                slips.append(float(np.linalg.norm(xy - prev_xy[name]) / dt))

            prev_xy[name] = xy if contacts[t].get(name, False) else None

    if not slips:
        return 0.0

    return float(np.mean(slips))

"""
Headless test harness for the OpenMBD verification benchmark suite.
====================================================================
`build_engine()` loads a JSON model and constructs a PhysicsEngine exactly
as the GUI does (SimulatorGUI.rebuild_physics), but with no tkinter
dependency.

WHY THIS FILE IS MORE THAN A LOADER
-----------------------------------
`ModelConfig.joints` / `ModelConfig.joint_vels` are GUI-facing structures:
three numbers per joint, labelled in Z,Y,X order. How `physics_engine.py`
maps those three numbers onto a joint's actual degree(s) of freedom has
changed more than once during development, and a benchmark that hard-codes
one mapping silently stops testing what it claims to test when the mapping
changes. Two concrete cases seen in this project:

  * The free root joint's angular velocity is reversed internally in some
    builds ([wZ,wY,wX] -> (wx,wy,wz)) and not in others. Benchmark 1 was
    handed omega=(5, 0.3, 0.2) rad/s and integrated (0.2, 0.3, 5.0) -- a
    completely different tumble, which showed up as a 178 deg "orientation
    failure" that was dt-independent (i.e. not truncation error at all).

  * A revolute joint's angle is read from whichever ZYX slot the dominant
    component of `joint_axis_local` points at. Once `joint_axis_local` was
    corrected to be the CHILD-frame axis (always (0,0,1)), that rule began
    selecting slot 2 for every revolute joint in every model. A benchmark
    passing its deflection in slot 0 therefore had BOTH its initial angle
    and its initial rate silently replaced by zero -- and a multibody
    system released from rest in uniform gravity provably shows zero
    internal motion, so Benchmarks 2 and 4 still "PASSED" their
    conservation checks while testing nothing but rigid free fall.

This harness therefore states initial conditions PHYSICALLY and makes the
engine agree, whatever its internal convention:

  * angular velocities of the root (dof=7) and spherical (dof=4) joints are
    given in (x, y, z) order; the harness reads back what the engine stored
    and, if the engine reversed it, rebuilds once with the input
    pre-reversed;
  * revolute joints are given as SCALARS -- one angle in degrees, one rate
    in rad/s -- and the harness probes which ZYX slot this build reads
    each of them from, then fills that slot.

Both mechanisms are exact (direct comparison of requested vs. stored
state), not heuristics. `build_engine(..., verify=True)` raises if the
requested state could not be reproduced, so a future convention change
becomes a loud error instead of a quiet false PASS. The resolved
conventions are reported on `engine.benchmark_init_report` for logging.
"""
import numpy as np
from physics_engine import PhysicsEngine
from physics_rigid_body import RigidBody
from models_parser import JSONModelParser
from models_multibody import MultibodyHumanModel
from models_config import ModelConfig


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------

def _construct(json_path, pos_str, vel_str, joints, joint_vels, dt):
    cfg = ModelConfig(0)
    cfg.path = json_path
    cfg.pos_str = pos_str
    cfg.vel_str = vel_str
    cfg.joints = joints or {}
    cfg.joint_vels = joint_vels or {}
    cfg.joint_torques = {}

    cfg.parser = JSONModelParser(json_path)
    cfg.multibody_model = MultibodyHumanModel(json_path)
    cfg.is_loaded = True

    model_bodies = []
    for name, data in cfg.parser.bodies.items():
        rb = RigidBody(f"0_{name}", data, 0)
        model_bodies.append(rb)

    engine = PhysicsEngine()
    engine.dt = dt
    engine.add_model(cfg.multibody_model, model_bodies, cfg)

    # add_model() populates body.pos/R (via _record_initial_overlaps ->
    # update_kinematics_from_q) but body.vel/ang_vel are only ever set
    # inside step() -> _update_body_velocities_from_qdot(). Populate them
    # now so callers can read a fully-consistent t=0 state before the
    # first integration step.
    q0 = engine.state[:engine.nq]
    qdot0 = engine.state[engine.nq:]
    engine.update_kinematics_from_q(q0)
    engine._update_body_velocities_from_qdot(q0, qdot0)
    return engine


def _joint_slots(engine):
    """{joint_name: (start_idx, dof, is_root, type)} for every joint."""
    out = {}
    for (midx, jname, jinfo, dof) in engine.joint_list:
        s, _ = engine.joint_dof_map[(midx, jname)]
        out[jname] = (s, dof, bool(jinfo.get('is_root_joint', False)),
                      jinfo.get('type', 'fixed'))
    return out


def _resolve(name, slots, root_name):
    """Map the caller's key ('root_joint' alias included) to a real joint."""
    if name in slots:
        return name
    if name == 'root_joint' and root_name is not None:
        return root_name
    return None


def _as_scalar(value):
    """Accept a scalar or a list; a list contributes its largest |element|."""
    if np.isscalar(value):
        return float(value)
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr[int(np.argmax(np.abs(arr)))])


# ----------------------------------------------------------------------
# convention probing
# ----------------------------------------------------------------------

_PROBE_ANG = np.array([11.0, 23.0, 37.0])      # deg, mutually distinguishable
_PROBE_VEL = np.array([1.7, 2.9, 4.3])         # rad/s


def _probe_revolute_slots(json_path, pos_str, vel_str, dt, revolute_names):
    """
    Determine, for this engine build, which ZYX slot a revolute joint's
    angle and rate are actually read from. Builds one throwaway engine with
    three distinguishable probe values per joint and matches the resulting
    q[s] / qdot[s] back to the slot that produced them.
    """
    probe_joints = {n: list(_PROBE_ANG) for n in revolute_names}
    probe_vels = {n: list(_PROBE_VEL) for n in revolute_names}
    eng = _construct(json_path, pos_str, vel_str, probe_joints, probe_vels, dt)
    slots = _joint_slots(eng)

    ang_slot, vel_slot = {}, {}
    for n in revolute_names:
        if n not in slots:
            continue
        s, dof, _, _ = slots[n]
        if dof != 1:
            continue
        got_ang = np.degrees(float(eng.state[s]))
        got_vel = float(eng.state[eng.nq + s])
        a_match = np.argmin(np.abs(_PROBE_ANG - got_ang))
        v_match = np.argmin(np.abs(_PROBE_VEL - got_vel))
        ang_slot[n] = (int(a_match)
                       if abs(_PROBE_ANG[a_match] - got_ang) < 1e-6 else None)
        vel_slot[n] = (int(v_match)
                       if abs(_PROBE_VEL[v_match] - got_vel) < 1e-9 else None)
    return ang_slot, vel_slot


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------

def build_engine(json_path, pos_str="0 0 0", vel_str="0 0 0",
                 joints=None, joint_vels=None, dt=1e-5, verify=True):
    """
    Build a PhysicsEngine from a JSON model with physically-stated ICs.

    joints      : {joint_name: value}
                  * revolute joint -> scalar angle in DEGREES (a 3-list is
                    accepted for backward compatibility; its largest-
                    magnitude element is taken as the intended angle)
                  * spherical joint -> [Z, Y, X] Euler degrees, as before
    joint_vels  : {joint_name: value}
                  * root (dof=7) / spherical (dof=4) -> angular velocity as
                    a 3-vector in PHYSICAL (x, y, z) order
                  * revolute -> scalar rate in rad/s (3-list also accepted)
    verify      : raise if the engine state does not match the request.
    """
    joints = dict(joints or {})
    joint_vels = dict(joint_vels or {})

    # --- one cheap build to discover the model's joint inventory --------
    probe = _construct(json_path, pos_str, vel_str, {}, {}, dt)
    slots = _joint_slots(probe)
    root_name = next((n for n, (s, d, r, t) in slots.items() if r and d == 7),
                     None)

    revolute_keys = []
    for key in set(list(joints) + list(joint_vels)):
        name = _resolve(key, slots, root_name)
        if name is not None and slots[name][1] == 1 and slots[name][3] == 'revolute':
            revolute_keys.append((key, name))

    # --- resolve the revolute ZYX slot convention for this build --------
    ang_slot, vel_slot = ({}, {})
    if revolute_keys:
        ang_slot, vel_slot = _probe_revolute_slots(
            json_path, pos_str, vel_str, dt, [n for _, n in revolute_keys])

    eng_joints, eng_vels = dict(joints), dict(joint_vels)
    for key, name in revolute_keys:
        if key in joints:
            a = _as_scalar(joints[key])
            vec = [0.0, 0.0, 0.0]
            vec[ang_slot.get(name) or 0] = a
            eng_joints[key] = vec
        if key in joint_vels:
            v = _as_scalar(joint_vels[key])
            vec = [0.0, 0.0, 0.0]
            vec[vel_slot.get(name) or 0] = v
            eng_vels[key] = vec

    eng = _construct(json_path, pos_str, vel_str, eng_joints, eng_vels, dt)

    # --- angular-velocity component order (root / spherical) ------------
    def angular_entries(engine):
        sl = _joint_slots(engine)
        nq = engine.nq
        for key in joint_vels:
            name = _resolve(key, sl, root_name)
            if name is None:
                continue
            s, dof, is_root, _ = sl[name]
            if is_root and dof == 7:
                yield key, slice(nq + s + 3, nq + s + 6)
            elif dof == 4:
                yield key, slice(nq + s, nq + s + 3)

    reversed_keys = []
    for key, sl_ in angular_entries(eng):
        want = np.asarray(joint_vels[key][:3], dtype=float)
        got = np.asarray(eng.state[sl_], dtype=float)
        if np.allclose(got, want, atol=1e-12):
            continue
        if np.allclose(got, want[::-1], atol=1e-12):
            reversed_keys.append(key)

    if reversed_keys:
        for key in reversed_keys:
            v = list(joint_vels[key])
            eng_vels[key] = list(np.asarray(v[:3], dtype=float)[::-1]) + list(v[3:])
        eng = _construct(json_path, pos_str, vel_str, eng_joints, eng_vels, dt)

    # --- verification ---------------------------------------------------
    report = {'angvel_order_reversed_by_engine': reversed_keys,
              'revolute_angle_slot': dict(ang_slot),
              'revolute_rate_slot': dict(vel_slot),
              'mismatch': []}

    for key, sl_ in angular_entries(eng):
        want = np.asarray(joint_vels[key][:3], dtype=float)
        got = np.asarray(eng.state[sl_], dtype=float)
        if not np.allclose(got, want, atol=1e-10):
            report['mismatch'].append(('angvel', key, want.tolist(), got.tolist()))

    sl_final = _joint_slots(eng)
    for key, name in revolute_keys:
        s = sl_final[name][0]
        if key in joints:
            want = np.radians(_as_scalar(joints[key]))
            got = float(eng.state[s])
            if abs(got - want) > 1e-9:
                report['mismatch'].append(
                    ('angle', key, float(np.degrees(want)), float(np.degrees(got))))
        if key in joint_vels:
            want = _as_scalar(joint_vels[key])
            got = float(eng.state[eng.nq + s])
            if abs(got - want) > 1e-9:
                report['mismatch'].append(('rate', key, want, got))

    eng.benchmark_init_report = report

    if verify and report['mismatch']:
        raise RuntimeError(
            "build_engine(): the engine did not accept the requested initial "
            f"conditions: {report['mismatch']}. This is a joint-convention "
            "mismatch between harness.py and physics_engine.py, not a "
            "dynamics error -- see the module docstring.")

    return eng


def describe_init(engine):
    """One-line summary of how the engine's conventions were resolved."""
    r = getattr(engine, 'benchmark_init_report', None)
    if not r:
        return "init: (no report)"
    bits = []
    if r['angvel_order_reversed_by_engine']:
        bits.append("root/spherical omega order reversed by engine "
                    f"{r['angvel_order_reversed_by_engine']}")
    if r['revolute_angle_slot']:
        bits.append("revolute ZYX slots (angle/rate): " + ", ".join(
            f"{k}={r['revolute_angle_slot'].get(k)}/{r['revolute_rate_slot'].get(k)}"
            for k in r['revolute_angle_slot']))
    return "init conventions resolved: " + ("; ".join(bits) if bits else "defaults")


def system_conservation_metrics(bodies, masses, g=9.81):
    """
    Returns (cg, KE, PE, L_about_cg) for a set of RigidBody objects.

    L_about_cg = sum_i [ (r_i - R_cg) x m_i v_i  +  I_i_world @ omega_i ]

    For a system under uniform gravity with only internal (Newton's-3rd-law)
    joint forces/torques, L_about_cg is conserved EXACTLY: gravity is a
    uniform body force, so its net torque about the system's own centre of
    mass is identically zero (sum_i m_i (r_i - R_cg) = 0 by definition of
    the CG), and internal joint forces/torques cancel in the system total.
    """
    m_tot = sum(masses)
    cg = sum(m * b.pos for b, m in zip(bodies, masses)) / m_tot
    KE = 0.0
    PE = 0.0
    L = np.zeros(3)
    for b, m in zip(bodies, masses):
        KE += b.get_kinetic_energy()
        PE += m * g * b.pos[2]
        Iw = b.get_world_inertia()
        L += np.cross(b.pos - cg, m * b.vel) + Iw @ b.ang_vel
    return cg, KE, PE, L


def body_by_name(engine, short_name):
    for b in engine.bodies:
        if b.name == f"0_{short_name}":
            return b
    raise KeyError(short_name)

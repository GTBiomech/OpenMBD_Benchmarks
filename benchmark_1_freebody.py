"""
Benchmark 1 -- Free rigid body under gravity
=============================================
Exercises: RNEA bias-force assembly for the 7-DOF free root joint,
analytic Jacobians A1/A2 (translational + rotational), the mass-matrix
solve, and the quaternion time-integration scheme in physics_engine.step().

No contact geometry is attached to the body, so the ONLY force acting is
gravity (through the CG), which produces zero torque about the CG. The
system therefore has two independent, closed-form/known-ODE ground truths:

  1. Translational motion is exact projectile motion.
  2. Rotational motion is torque-free rigid-body rotation, governed by the
     classical Euler equations, integrated here completely independently
     (via SciPy, not via OpenMBD) to serve as ground truth.

Both are compared against the trajectory produced by the actual
physics_engine.PhysicsEngine.
"""
import numpy as np
from scipy.integrate import solve_ivp
from harness import build_engine, body_by_name

G = 9.81
I1, I2, I3 = 0.03, 0.05, 0.08          # must match bench_freebody.json
MASS = 2.0

POS0 = np.array([0.0, 0.0, 5.0])
VEL0 = np.array([3.0, 0.0, 4.0])
OMEGA0_BODY = np.array([5.0, 0.3, 0.2])  # rad/s, body frame == world frame at t=0 (quat=I)
DT = 1e-5
DURATION = 1.0


def quat_mult(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def reference_rhs(t, y):
    w1, w2, w3, qw, qx, qy, qz = y
    dw1 = (I2 - I3) / I1 * w2 * w3
    dw2 = (I3 - I1) / I2 * w3 * w1
    dw3 = (I1 - I2) / I3 * w1 * w2
    dq = 0.5 * quat_mult([qw, qx, qy, qz], [0.0, w1, w2, w3])
    return [dw1, dw2, dw3, dq[0], dq[1], dq[2], dq[3]]


def quat_angle_diff_deg(q_ref, q_sim):
    """Angle (deg) of the relative rotation between two unit quaternions."""
    q_ref = q_ref / np.linalg.norm(q_ref)
    q_sim = q_sim / np.linalg.norm(q_sim)
    dot = np.clip(abs(np.dot(q_ref, q_sim)), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def run(dt=DT, duration=DURATION, verbose=True):
    # ---- Independent reference solution (SciPy, high accuracy) ---------
    y0 = [*OMEGA0_BODY, 1.0, 0.0, 0.0, 0.0]
    t_eval = np.arange(0.0, duration + 1e-9, 0.01)
    sol = solve_ivp(reference_rhs, [0.0, duration], y0, t_eval=t_eval,
                     method='DOP853', rtol=1e-12, atol=1e-13)
    ref_t = sol.t
    ref_omega_body = sol.y[0:3, :].T
    ref_quat = sol.y[3:7, :].T
    ref_quat /= np.linalg.norm(ref_quat, axis=1, keepdims=True)

    ref_pos = POS0[None, :] + VEL0[None, :] * ref_t[:, None]
    ref_pos[:, 2] += -0.5 * G * ref_t ** 2
    ref_vel = np.tile(VEL0, (len(ref_t), 1))
    ref_vel[:, 2] += -G * ref_t

    # ---- OpenMBD engine run ---------------------------------------------
    eng = build_engine('bench_freebody.json',
                        pos_str=f"{POS0[0]} {POS0[1]} {POS0[2]}",
                        vel_str=f"{VEL0[0]} {VEL0[1]} {VEL0[2]}",
                        joint_vels={'root_joint': list(OMEGA0_BODY)},
                        dt=dt)
    body = body_by_name(eng, 'body1')

    n_steps = int(round(duration / dt))
    sample_every = max(1, int(round(0.01 / dt)))

    sim_t, sim_pos, sim_vel, sim_quat, sim_omega_body = [], [], [], [], []
    sim_t.append(0.0); sim_pos.append(body.pos.copy()); sim_vel.append(body.vel.copy())
    sim_quat.append(body.quat.copy()); sim_omega_body.append(body.R.T @ body.ang_vel)

    for step in range(1, n_steps + 1):
        eng.step()
        if step % sample_every == 0:
            sim_t.append(eng.time)
            sim_pos.append(body.pos.copy())
            sim_vel.append(body.vel.copy())
            sim_quat.append(body.quat.copy())
            sim_omega_body.append(body.R.T @ body.ang_vel)

    sim_t = np.array(sim_t); sim_pos = np.array(sim_pos); sim_vel = np.array(sim_vel)
    sim_quat = np.array(sim_quat); sim_omega_body = np.array(sim_omega_body)

    n = min(len(sim_t), len(ref_t))
    pos_err = np.linalg.norm(sim_pos[:n] - ref_pos[:n], axis=1)
    vel_err = np.linalg.norm(sim_vel[:n] - ref_vel[:n], axis=1)
    omega_err = np.linalg.norm(sim_omega_body[:n] - ref_omega_body[:n], axis=1)
    quat_err_deg = np.array([quat_angle_diff_deg(ref_quat[i], sim_quat[i]) for i in range(n)])

    pos_scale = np.linalg.norm(ref_pos[n-1] - ref_pos[0])
    tol_pos = 1e-3 * max(pos_scale, 1.0)
    tol_vel = 1e-3 * max(np.linalg.norm(ref_vel).max(), 1.0)
    tol_omega = 1e-2
    tol_quat_deg = 0.5
    passed = (pos_err.max() < tol_pos and vel_err.max() < tol_vel and
              omega_err.max() < tol_omega and quat_err_deg.max() < tol_quat_deg)

    if verbose:
        print("=" * 70)
        print("BENCHMARK 1: Free rigid body under gravity")
        print("=" * 70)
        print(f"dt = {dt}, duration = {duration} s, {n_steps} integration steps")
        print(f"max |position error|      : {pos_err.max():.3e} m   "
              f"(final: {pos_err[-1]:.3e} m)")
        print(f"max |velocity error|      : {vel_err.max():.3e} m/s "
              f"(final: {vel_err[-1]:.3e} m/s)")
        print(f"max |omega_body error|    : {omega_err.max():.3e} rad/s "
              f"(final: {omega_err[-1]:.3e} rad/s)")
        print(f"max quaternion angle error: {quat_err_deg.max():.3e} deg "
              f"(final: {quat_err_deg[-1]:.3e} deg)")
        print("-" * 70)
        print(f"Tolerances: pos<{tol_pos:.2e} m, vel<{tol_vel:.2e} m/s, "
              f"omega<{tol_omega:.2e} rad/s, quat<{tol_quat_deg:.2e} deg")
        print("RESULT:", "PASS" if passed else "FAIL")
        print("=" * 70)

    return dict(t=sim_t[:n], pos_err=pos_err, vel_err=vel_err,
                omega_err=omega_err, quat_err_deg=quat_err_deg,
                sim_pos=sim_pos[:n], ref_pos=ref_pos[:n],
                sim_omega=sim_omega_body[:n], ref_omega=ref_omega_body[:n],
                passed=passed)


if __name__ == '__main__':
    run()

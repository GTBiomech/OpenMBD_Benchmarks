"""
Benchmark 3 -- Two-body chain with a SPHERICAL (ball) joint
============================================================
Same conservation-law strategy as Benchmark 2, but the child joint is a
spherical joint carried as a unit quaternion (dof=4), which is an entirely
separate code path from Benchmark 2's revolute (dof=1) joint:

  * state/velocity initialisation  (dof==4 branch of
    _initialize_state_from_config)
  * Jacobian columns               (dof==4 branch of compute_a1_a2_analytic,
                                    where A2's columns are the child's own
                                    rotation-matrix columns, since qdot holds
                                    a LOCAL-frame omega)
  * quaternion time integration    (dof==4 branch of step(), which uses the
                                    RIGHT/body-frame Hamilton product -- the
                                    correct one for a local omega, and the
                                    one that was wrongly being used for the
                                    world-frame root joint before the fix
                                    reported in Benchmark 1)

Checked against three exact conservation laws (no closed-form trajectory
exists for this coupled nonlinear system):

  1. Momentum  -- the whole-system centre of mass must follow an exact
     projectile parabola (internal joint forces cancel pairwise).
  2. Energy    -- with the generic passive joint damping disabled
     (engine.JOINT_PASSIVE_D = 0) the system is conservative.
  3. Angular momentum -- total L about the *moving* system CG is conserved
     exactly: uniform gravity exerts zero net torque about the system's own
     centre of mass, and internal joint torques cancel. This check is only
     meaningful with a genuinely 3-D angular velocity, which a 1-DOF
     revolute joint cannot produce, so it is specific to this benchmark.

As in Benchmarks 2 and 4, the joint is given a nonzero initial ANGULAR
VELOCITY rather than a static deflection: a system released from rest in a
uniform gravitational field provably shows zero internal motion.
"""
import numpy as np
from harness import (build_engine, body_by_name, describe_init,
                     system_conservation_metrics)

G = 9.81
MASS_BASE, MASS_ARM = 2.5, 0.8
DT = 1e-5
DURATION = 0.8
OMEGA0 = [2.0, -1.5, 1.0]     # rad/s, child-local frame, fully 3-D


def run(dt=DT, duration=DURATION, verbose=True):
    eng = build_engine('bench_spherical.json', pos_str="0 0 3.0", vel_str="0 0 0",
                       joint_vels={'ball_joint': OMEGA0}, dt=dt)
    eng.JOINT_PASSIVE_D = 0.0      # isolate a conservative system
    names = ['base', 'arm']
    bodies = [body_by_name(eng, n) for n in names]
    masses = [MASS_BASE, MASS_ARM]

    cg0, KE0, PE0, L0 = system_conservation_metrics(bodies, masses, G)
    v_cg0 = sum(m * b.vel for b, m in zip(bodies, masses)) / sum(masses)
    E0 = KE0 + PE0
    L0_norm = np.linalg.norm(L0)

    n_steps = int(round(duration / dt))
    sample_every = max(1, int(round(0.005 / dt)))

    t_hist, cg_hist, E_hist, L_hist = [0.0], [cg0.copy()], [E0], [L0.copy()]
    for step in range(1, n_steps + 1):
        eng.step()
        if step % sample_every == 0:
            cg, KE, PE, L = system_conservation_metrics(bodies, masses, G)
            t_hist.append(eng.time)
            cg_hist.append(cg.copy())
            E_hist.append(KE + PE)
            L_hist.append(L.copy())

    t_hist = np.array(t_hist)
    cg_hist = np.array(cg_hist)
    E_hist = np.array(E_hist)
    L_hist = np.array(L_hist)

    cg_exact = cg0[None, :] + v_cg0[None, :] * t_hist[:, None]
    cg_exact[:, 2] += -0.5 * G * t_hist ** 2
    cg_err = np.linalg.norm(cg_hist - cg_exact, axis=1)
    E_drift = np.abs(E_hist - E0) / abs(E0)
    L_drift = np.linalg.norm(L_hist - L0[None, :], axis=1) / max(L0_norm, 1e-12)

    if verbose:
        print("=" * 70)
        print("BENCHMARK 3: Two-body chain (free base + SPHERICAL/quaternion joint)")
        print("=" * 70)
        print(f"dt = {dt}, duration = {duration} s, {n_steps} integration steps")
        print(describe_init(eng))
        print(f"masses: base={MASS_BASE} kg, arm={MASS_ARM} kg; "
              f"initial joint omega (local) = {OMEGA0} rad/s")
        print(f"[Momentum] max |system CG error| vs exact parabola : {cg_err.max():.3e} m")
        print(f"[Energy]   E(0) = {E0:.6f} J; max relative drift    : {E_drift.max():.3e}")
        print(f"[Ang.mom.] |L(0)| = {L0_norm:.6f} kg m^2/s; "
              f"max relative drift : {L_drift.max():.3e}")

    tol_cg, tol_E, tol_L = 1e-3, 1e-3, 1e-2
    passed = (cg_err.max() < tol_cg and E_drift.max() < tol_E
              and L_drift.max() < tol_L)
    if verbose:
        print("-" * 70)
        print(f"Tolerances: CG error < {tol_cg:.1e} m, energy drift < {tol_E:.1e}, "
              f"|L| drift < {tol_L:.1e} (rel.)")
        print("RESULT:", "PASS" if passed else "FAIL")
        print("=" * 70)

    return dict(t=t_hist, cg_err=cg_err, E_hist=E_hist, E0=E0,
                E_drift=E_drift, L_hist=L_hist, L0=L0, L_drift=L_drift,
                passed=passed)


if __name__ == '__main__':
    run()

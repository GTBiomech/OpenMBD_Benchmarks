"""
Benchmark 2 -- Two-body articulated chain (free base + revolute arm)
=====================================================================
Exercises: the RNEA backward pass across a joint (parent<-child force/
torque accumulation, Newton's-3rd-law-consistent joint reaction), the
revolute-joint Jacobian columns, and the coupling between a free (7-DOF)
root and a child joint in the assembled mass matrix.

No independent closed-form trajectory exists for this coupled nonlinear
system, so it is instead checked against two conservation laws that hold
EXACTLY, by Newton's laws, for ANY internal joint topology/parameters:

  1. Momentum: internal joint forces are equal-and-opposite (Newton's 3rd
     law), so they cancel in the SYSTEM total. The whole-system centre of
     mass must therefore follow an exact projectile-motion parabola under
     gravity alone, regardless of how the arm swings relative to the base.

  2. Energy: with the generic passive joint-damping term disabled
     (engine.JOINT_PASSIVE_D = 0), the system is conservative (no contact,
     no dissipation), so total mechanical energy (KE_base + KE_arm + PE)
     must be conserved over the simulation.

Both are necessary consequences of a correct RNEA/Jacobian/force-assembly
implementation; a bug in the joint force projection or backward pass would
generically break at least one of them.

NOTE: the arm is given a nonzero INITIAL ANGULAR VELOCITY, not just a static
initial deflection. A system released from REST in uniform gravity, connected
only by ideal joints, provably shows ZERO relative/internal motion (ONLY the
system's centre of mass free-falls; this is an exact consequence of the
equivalence principle -- transforming to the freely-falling frame cancels
gravity identically for every particle, leaving a force-free system that, if
it starts at rest, stays at rest). A "deflect and release" test would
therefore be blind to real dynamics (and to bugs in the revolute-joint axis
handling -- see physics_engine_fixes.patch) regardless of engine correctness;
an initial velocity is required to exercise genuine coupled dynamics.
"""
import numpy as np
from harness import build_engine, body_by_name, describe_init

G = 9.81
MASS_BASE, MASS_ARM = 3.0, 1.0
DT = 1e-5
DURATION = 0.8
INITIAL_DEFLECTION_DEG = 55.0
INITIAL_ANGULAR_VEL = 3.0   # rad/s -- see module docstring: a system released from
                            # REST in uniform gravity provably shows zero relative
                            # motion (an exact consequence of the equivalence
                            # principle for any ideal-joint multibody system), so an
                            # initial velocity is needed for a meaningful dynamics
                            # test, not just a conservation-law check on a system
                            # that never actually moves internally.


def system_state(base, arm):
    m_tot = MASS_BASE + MASS_ARM
    cg = (MASS_BASE * base.pos + MASS_ARM * arm.pos) / m_tot
    KE = base.get_kinetic_energy() + arm.get_kinetic_energy()
    PE = MASS_BASE * G * base.pos[2] + MASS_ARM * G * arm.pos[2]
    return cg, KE, PE


def run(dt=DT, duration=DURATION, verbose=True):
    # Initial conditions are stated PHYSICALLY (one angle, one rate about the
    # joint's own axis); harness.build_engine() resolves whichever ZYX slot
    # convention this engine build reads them from, and raises if they do not
    # reach the state vector -- see harness.py's docstring for why a benchmark
    # that skips that check can "PASS" while simulating rigid free fall.
    eng = build_engine('bench_twobody.json', pos_str="0 0 3.0", vel_str="0 0 0",
                        joints={'test_pivot': INITIAL_DEFLECTION_DEG},
                        joint_vels={'test_pivot': INITIAL_ANGULAR_VEL},
                        dt=dt)
    eng.JOINT_PASSIVE_D = 0.0   # isolate a conservative system for the energy check
    base = body_by_name(eng, 'base')
    arm = body_by_name(eng, 'arm')

    cg0, KE0, PE0 = system_state(base, arm)
    v_cg0 = (MASS_BASE * base.vel + MASS_ARM * arm.vel) / (MASS_BASE + MASS_ARM)
    E0 = KE0 + PE0
    m_tot = MASS_BASE + MASS_ARM

    n_steps = int(round(duration / dt))
    sample_every = max(1, int(round(0.005 / dt)))

    t_hist, cg_hist, E_hist = [0.0], [cg0.copy()], [E0]
    for step in range(1, n_steps + 1):
        eng.step()
        if step % sample_every == 0:
            cg, KE, PE = system_state(base, arm)
            t_hist.append(eng.time)
            cg_hist.append(cg.copy())
            E_hist.append(KE + PE)

    t_hist = np.array(t_hist)
    cg_hist = np.array(cg_hist)
    E_hist = np.array(E_hist)

    # ---- Check 1: system CG must be exact projectile motion -------------
    cg_exact = cg0[None, :] + v_cg0[None, :] * t_hist[:, None]
    cg_exact[:, 2] += -0.5 * G * t_hist ** 2
    cg_err = np.linalg.norm(cg_hist - cg_exact, axis=1)

    # ---- Check 2: total mechanical energy conservation -------------------
    E_drift = np.abs(E_hist - E0) / abs(E0)

    if not verbose:
        return dict(t=t_hist, cg_err=cg_err, E_hist=E_hist, E0=E0,
                    E_drift=E_drift,
                    passed=(cg_err.max() < 1e-3 and E_drift.max() < 1e-3))

    print("=" * 70)
    print("BENCHMARK 2: Two-body chain (free base + revolute arm)")
    print("=" * 70)
    print(f"dt = {dt}, duration = {duration} s, {n_steps} integration steps")
    print(describe_init(eng))
    print(f"masses: base={MASS_BASE} kg, arm={MASS_ARM} kg  "
          f"(total {m_tot} kg); initial arm deflection {INITIAL_DEFLECTION_DEG} deg")
    print(f"[Momentum] max |system CG error| vs exact parabola : "
          f"{cg_err.max():.3e} m  (final: {cg_err[-1]:.3e} m)")
    print(f"[Energy]   max relative |E(t)-E(0)|/|E(0)|          : "
          f"{E_drift.max():.3e}  (final: {E_drift[-1]:.3e})")
    print(f"           E(0) = {E0:.6f} J")

    tol_cg = 1e-3      # metres, over a ~3 m fall
    tol_E = 1e-3       # 0.1% relative energy drift

    passed = cg_err.max() < tol_cg and E_drift.max() < tol_E
    print("-" * 70)
    print(f"Tolerances: CG error < {tol_cg:.1e} m, energy drift < {tol_E:.1e} (rel.)")
    print("RESULT:", "PASS" if passed else "FAIL")
    print("=" * 70)

    return dict(t=t_hist, cg_err=cg_err, E_hist=E_hist, E0=E0,
                E_drift=E_drift, passed=passed)


if __name__ == '__main__':
    run()

"""
Benchmark 4 -- Three-body serial chain (free base -> revolute -> revolute)
============================================================================
Benchmarks 2 and 3 each used exactly one child joint, so RNEA's backward
pass only ever had to combine a single child's (force, torque) into its
one parent (rnea()'s "f[self.parent_idx[i]] += f[i]" / "tau[...] += tau[i]
+ cross(r, f[i])", executed once). This benchmark chains TWO revolute
joints (base -> arm1 -> arm2) so that body "arm1" is BOTH a child (of the
free base) AND a parent (of arm2), requiring the recursion to correctly
accumulate arm2's reaction back through arm1 and on to the base.

Checked, as in Benchmark 2, against exact conservation laws (system-CG
projectile motion for momentum; total mechanical energy with passive
joint damping disabled) -- a bug in the multi-level recursion would
generically corrupt both.

Both joints are given nonzero initial angular velocities, not just static
deflections -- see Benchmark 2's docstring: a system released from rest in
uniform gravity provably shows zero relative motion, so this is required
for a meaningful test.
"""
import numpy as np
from harness import (build_engine, body_by_name, describe_init,
                     system_conservation_metrics)

G = 9.81
MASSES = {'base': 4.0, 'arm1': 1.0, 'arm2': 0.6}
DT = 1e-5
DURATION = 0.8


def run(dt=DT, duration=DURATION, verbose=True):
    # Physically-stated ICs; see benchmark_2_twobody.py / harness.py.
    eng = build_engine('bench_threebody.json', pos_str="0 0 3.0", vel_str="0 0 0",
                        joints={'pivot1': 50.0, 'pivot2': -35.0},
                        joint_vels={'pivot1': 2.0, 'pivot2': -1.5},
                        dt=dt)
    eng.JOINT_PASSIVE_D = 0.0
    names = ['base', 'arm1', 'arm2']
    bodies = [body_by_name(eng, n) for n in names]
    masses = [MASSES[n] for n in names]

    cg0, KE0, PE0, L0 = system_conservation_metrics(bodies, masses, G)
    v_cg0 = sum(m * b.vel for b, m in zip(bodies, masses)) / sum(masses)
    E0 = KE0 + PE0

    n_steps = int(round(duration / dt))
    sample_every = max(1, int(round(0.005 / dt)))

    t_hist, cg_hist, E_hist = [0.0], [cg0.copy()], [E0]
    for step in range(1, n_steps + 1):
        eng.step()
        if step % sample_every == 0:
            cg, KE, PE, L = system_conservation_metrics(bodies, masses, G)
            t_hist.append(eng.time)
            cg_hist.append(cg.copy())
            E_hist.append(KE + PE)

    t_hist = np.array(t_hist)
    cg_hist = np.array(cg_hist)
    E_hist = np.array(E_hist)

    cg_exact = cg0[None, :] + v_cg0[None, :] * t_hist[:, None]
    cg_exact[:, 2] += -0.5 * G * t_hist ** 2
    cg_err = np.linalg.norm(cg_hist - cg_exact, axis=1)
    E_drift = np.abs(E_hist - E0) / abs(E0)

    if not verbose:
        return dict(t=t_hist, cg_err=cg_err, E_hist=E_hist, E0=E0,
                    E_drift=E_drift,
                    passed=(cg_err.max() < 1e-3 and E_drift.max() < 1e-3))

    print("=" * 70)
    print("BENCHMARK 4: Three-body chain (free base -> revolute -> revolute)")
    print("=" * 70)
    print(f"dt = {dt}, duration = {duration} s, {n_steps} integration steps")
    print(describe_init(eng))
    print(f"masses: base={MASSES['base']}, arm1={MASSES['arm1']}, arm2={MASSES['arm2']} kg "
          f"(total {sum(masses)} kg)")
    print(f"[Momentum] max |system CG error| vs exact parabola : {cg_err.max():.3e} m")
    print(f"[Energy]   E(0) = {E0:.6f} J; max relative |E(t)-E(0)|/E(0) = {E_drift.max():.3e}")

    tol_cg = 1e-3
    tol_E = 1e-3
    passed = cg_err.max() < tol_cg and E_drift.max() < tol_E
    print("-" * 70)
    print(f"Tolerances: CG error < {tol_cg:.1e} m, energy drift < {tol_E:.1e} (rel.)")
    print("RESULT:", "PASS" if passed else "FAIL")
    print("=" * 70)

    return dict(t=t_hist, cg_err=cg_err, E_hist=E_hist, E0=E0,
                E_drift=E_drift, passed=passed)


if __name__ == '__main__':
    run()

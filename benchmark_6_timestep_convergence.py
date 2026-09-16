"""
Benchmark 6 -- Timestep convergence study (reviewer comment #6)
==================================================================
"The simulator uses a fixed-step Symplectic Euler method with a default
timestep of 0.0001 s. However, stiff contact and joint-limit penalties may
lead to timestep-dependent results. Please demonstrate timestep convergence
for representative simulations and report the effect of timestep size on
peak contact forces, trajectories, and energy behaviour."

This study swept dt from 8e-4 down to 1e-5 across three dynamic scenarios
(A, B, D below), each against its own independent ground truth (no OpenMBD
code involved in any reference), and separately swept contact STIFFNESS
against the values actually present in the bundled models (C). No error
metric plateaus at any particular dt -- accuracy keeps improving as dt
shrinks, at proportionally higher compute cost -- so the sweep instead
answers: "how many integration steps resolve one contact event at the
stiffness this project actually uses, and what dt does that require?"
Part C answers that directly: local slopes of the bundled force_curve
tables range from ~1e5 N/m (bicycle frame) to ~5-6e6 N/m (car floor/
wheel-hub contacts), and because contact duration scales as 1/sqrt(k),
dt=1e-4 s resolves the softest of these well (>100 steps/contact) but the
stiffest with only ~13-14 steps/contact and 0.3-0.6% peak-penetration
error per event -- comparable to how badly dt=8e-4 s resolves the softer
case in Part B. dt=1e-5 s restores >100 steps/contact and sub-0.05% error
across that range.

**PhysicsEngine's default dt, and the GUI's pre-filled Sim-Dt field, are
now 1e-5 s (changed from 1e-4 s) based on this result** -- see
physics_engine.py, ui_setup_tab.py; patch: default_timestep_1e-5.patch.

Four scenarios:
  A) Benchmark 1's tumbling free rigid body (smooth, non-contact dynamics;
     reference = independently-integrated Euler rigid-body equations).
  B) Benchmark 5's bouncing ball at k=5e4 N/m, lossless (reference = exact
     peak penetration from energy conservation).
  C) The same bouncing-ball setup, swept over contact stiffnesses spanning
     the range actually present in openmbd_car.json, openmbd_male.json,
     and openmbd_bicycle.json's force_curve tables.
  D) Benchmark 5b's bouncing ball at k=5e4 N/m WITH damping (c=70 N.s/m):
     converges more slowly than B at the same k, since damping adds a
     second, less dt-sensitive error source (see benchmark_5b_damped_contact.py).

Note on metric design (see benchmark_5b_damped_contact.py for detail): the
Part D force-error metric excludes the first ~1 ms of each contact event.
F=kx+c*xdot is mathematically discontinuous in time at first touch (jumps
to c*v_impact instantly, since the damping term depends on velocity, not
penetration); a discrete-time simulator needs ~1 step to register this,
and an earlier version of this sweep included that single-step transient
inconsistently (it only showed up for dt values whose 0.5 ms output-
sampling interval happened to land on the raw onset step), producing a
misleading, non-monotonic table. Excluding it gives a metric that reflects
genuine mid-contact tracking accuracy consistently across all dt.
"""
import numpy as np
import json
from harness import build_engine, body_by_name
import benchmark_1_freebody as b1
import benchmark_5_bouncingball as b5
import benchmark_5b_damped_contact as b5b

DTS = [8e-4, 4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1e-5]


def sweep_rotation():
    print("=" * 84)
    print("A) Tumbling free rigid body (Benchmark 1 scenario), T=0.2 s")
    print("=" * 84)
    print(f"{'dt (s)':>10} {'steps':>8} {'max pos err (m)':>16} {'max quat err (deg)':>20}")
    rows = []
    for dt in DTS:
        r = b1.run(dt=dt, duration=0.2, verbose=False)
        n_steps = int(round(0.2 / dt))
        print(f"{dt:10.2e} {n_steps:8d} {r['pos_err'].max():16.3e} {r['quat_err_deg'].max():20.4e}")
        rows.append((dt, r['pos_err'].max(), r['quat_err_deg'].max()))
    return rows


def sweep_contact():
    print()
    print("=" * 84)
    print("B) Bouncing ball, lossless contact (Benchmark 5 scenario, k=5e4 N/m)")
    print("=" * 84)

    _, _, t_impact, v_impact, t_contact_end_ref, v_rebound_ref = \
        b5.closed_form_trajectory(np.array([0.0]))
    k, m, g, R = b5.K, b5.MASS, b5.G, b5.RADIUS
    x_max_ref = (m * g + np.sqrt((m * g) ** 2 + k * m * v_impact ** 2)) / k
    F_peak_ref = k * x_max_ref
    contact_dur_ref = t_contact_end_ref - t_impact

    print(f"exact: peak penetration = {x_max_ref*1000:.4f} mm, "
          f"peak force = {F_peak_ref:.2f} N, contact duration = "
          f"{contact_dur_ref*1000:.4f} ms, restitution = 1.0 (lossless)")
    print(f"{'dt (s)':>10} {'steps/contact':>14} {'pen err (mm)':>14} "
          f"{'F_peak err (N)':>16} {'dur err (ms)':>14} {'|e-1|':>10}")

    rows = []
    for dt in DTS:
        eng = build_engine('bench_ball.json', pos_str=f"0 0 {b5.Z0}", vel_str="0 0 0", dt=dt)
        eng.contact_damping = 0.0
        ball = body_by_name(eng, 'ball')

        n_steps = int(round(1.0 / dt))
        min_z = ball.pos[2]
        t_start = t_end = None
        v_after = None
        was_touching = False
        for step in range(n_steps):
            eng.step()
            z = ball.pos[2]
            touching = z < R
            if touching and not was_touching:
                t_start = eng.time
            if (not touching) and was_touching:
                t_end = eng.time
                v_after = ball.vel[2]
            was_touching = touching
            if touching:
                min_z = min(min_z, z)
            if t_end is not None:
                break

        x_max_sim = R - min_z
        F_peak_sim = k * x_max_sim
        dur_sim = (t_end - t_start) if (t_start and t_end) else float('nan')
        e_sim = (v_after / v_impact) if v_after else float('nan')

        n_contact_steps = int(round(contact_dur_ref / dt))
        pen_err_mm = abs(x_max_sim - x_max_ref) * 1000
        F_err = abs(F_peak_sim - F_peak_ref)
        dur_err_ms = abs(dur_sim - contact_dur_ref) * 1000
        e_err = abs(e_sim - 1.0)

        print(f"{dt:10.2e} {n_contact_steps:14d} {pen_err_mm:14.4f} "
              f"{F_err:16.3f} {dur_err_ms:14.4f} {e_err:10.4f}")
        rows.append((dt, pen_err_mm, F_err, dur_err_ms, e_err))
    return rows


def sweep_damped_contact():
    print()
    print("=" * 84)
    print("D) Bouncing ball, DAMPED contact (Benchmark 5b scenario, k=5e4 N/m, c=70 N.s/m)")
    print("=" * 84)
    print(f"{'dt (s)':>10} {'steps/contact':>14} {'z err, contact (m)':>20} "
          f"{'F_peak err (N)':>16} {'F_peak err (%)':>15}")
    rows = []
    for dt in DTS:
        r = b5b.run(verbose=False, dt=dt)
        z_err = r['z_err_contact']
        F_err_pk = r['F_err_peak']
        n_contact_steps = int(round(r['contact_duration'] / dt))
        pct = F_err_pk / r['F_peak_ref'] * 100
        print(f"{dt:10.2e} {n_contact_steps:14d} {z_err:20.3e} {F_err_pk:16.3f} {pct:15.3f}")
        rows.append((dt, z_err, F_err_pk, pct))
    return rows


def compute_model_stiffnesses():
    """Recompute MODEL_STIFFNESSES from raw force_curve tables (for verification)."""
    curves = {
        'bicycle chain_stay': [[0,0],[0.001,100],[0.005,500],[0.01,1000],[0.02,2000],[0.05,5000],[0.1,10000]],
        'male pelvis_geom': [[0,0],[0.001,60],[0.005,600],[0.01,3000],[0.02,6500],[0.05,14000],[0.1,20000]],
        'car windscreen ws_base': [[0,0],[0.002,1200],[0.004,3500],[0.006,6500],[0.008,9500],[0.01,12500],[0.012,14500],[0.013,15000]],
        'car floor_main': [[0,0],[0.001,1000],[0.003,5000],[0.005,12000],[0.008,25000],[0.01,35000],[0.015,50000],[0.02,60000],[0.05,70000],[0.1,80000]],
        'car rim_FL (wheel hub)': [[0,0],[0.001,1500],[0.003,6000],[0.005,15000],[0.008,30000],[0.01,42000],[0.02,60000],[0.05,80000],[0.1,100000]],
    }
    out = {}
    for name, c in curves.items():
        slopes = [(c[i+1][1]-c[i][1])/(c[i+1][0]-c[i][0]) for i in range(len(c)-1)]
        out[name] = max(slopes)
    return out


MODEL_STIFFNESSES = compute_model_stiffnesses()


def sweep_stiffness(m=1.0, R=0.1, G=9.81):
    print()
    print("=" * 84)
    print("C) Same bouncing-ball contact, swept over stiffnesses ACTUALLY present")
    print("   in openmbd_car.json / openmbd_male.json / openmbd_bicycle.json")
    print("=" * 84)
    print("(contact duration scales as 1/sqrt(k): a 100x stiffer contact needs ~10x")
    print(" smaller dt to be resolved by the same number of steps -- this is why a")
    print(" single fixed dt cannot be simultaneously optimal for soft and stiff contacts)")
    print()
    v_impact = np.sqrt(2 * G * 0.9)   # same impact speed used throughout this suite

    rows = []
    for label, k in sorted(MODEL_STIFFNESSES.items(), key=lambda kv: kv[1]):
        model = {
            'name': 'k_sweep',
            'bodies': {'ball': {'mass': m, 'inertia': [0.004, 0.004, 0.004, 0, 0, 0],
                       'center_of_mass': [0, 0, 0],
                       'ellipsoids': [{'name': 'sphere', 'dimensions': [R, R, R],
                        'local_position': [0, 0, 0],
                        'local_orientation': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                        'force_curve': [[0.0, 0.0], [0.2, k*0.2]],
                        'unload_curve': [[0.0, 0.0], [0.2, k*0.2]]}]}},
            'joints': [{'name': 'root_joint', 'type': 'free', 'parent': 'GROUND', 'child': 'ball',
                        'T1': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                        'T2': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                        'T2_inv': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}],
        }
        with open('/tmp/k_sweep.json', 'w') as f:
            json.dump(model, f)

        x_max_ref = (m * G + np.sqrt((m * G) ** 2 + k * m * v_impact ** 2)) / k
        omega = np.sqrt(k / m)
        dur_est = np.pi / omega

        print(f"{label} (k~{k:.2e} N/m): omega={omega:.0f} rad/s, "
              f"contact duration~{dur_est*1000:.4f} ms")
        test_dts = [1e-4, 1e-5] if k < 2e6 else [1e-4, 1e-5, 1e-6]
        for dt in test_dts:
            eng = build_engine('/tmp/k_sweep.json', pos_str=f"0 0 {R}",
                                vel_str=f"0 0 {-v_impact}", dt=dt)
            eng.contact_damping = 0.0
            ball = body_by_name(eng, 'ball')
            T = 2.5 * dur_est
            n_steps = max(1, int(round(T / dt)))
            min_z = ball.pos[2]
            for _ in range(n_steps):
                eng.step()
                min_z = min(min_z, ball.pos[2])
            x_max_sim = R - min_z
            pen_err_pct = abs(x_max_sim - x_max_ref) / x_max_ref * 100
            steps_per_contact = dur_est / dt
            print(f"    dt={dt:.1e}  steps/contact~{steps_per_contact:8.1f}   "
                  f"peak-pen error = {pen_err_pct:7.3f} %")
            rows.append((label, k, dt, steps_per_contact, pen_err_pct))
        print()
    return rows


if __name__ == '__main__':
    rot_rows = sweep_rotation()
    con_rows = sweep_contact()
    damp_rows = sweep_damped_contact()
    stiff_rows = sweep_stiffness()

    print()
    print("=" * 84)
    print("Summary")
    print("=" * 84)
    print("Parts A/B/D show convergence (error shrinks, ~monotonically, as dt shrinks)")
    print("with no plateau at any particular dt for position/force/penetration metrics")
    print("-- i.e. accuracy keeps improving as dt shrinks, at proportionally higher")
    print("compute cost. Part D (damped contact) converges more slowly than Part B")
    print("(lossless, same k=5e4 N/m): going from dt=1e-4 to dt=1e-5 cuts its peak-")
    print("force error by only ~38% (2.9%->1.8%), versus >10x for Part B at the same")
    print("dt change -- k=5e4 N/m is already on the soft end of this project's own")
    print("stiffness range (see Part C), so there is less headroom left to recover.")
    print()
    print("Part C shows WHY dt can't be chosen independent of the model: this project's")
    print("own force_curve tables span local stiffnesses from ~1e5 to ~6e6 N/m (a 60x")
    print("range). At dt=1e-4, the softest of these (bicycle frame, ~1e5 N/m) is well")
    print("resolved (>100 steps/contact), but the stiffest (car floor/wheel contacts,")
    print("~5-6e6 N/m) gets only ~13-14 steps/contact -- comparable to the badly-under-")
    print("resolved dt=8e-4 case in Part B -- with 0.3-0.6% peak-penetration error per")
    print("event. dt=1e-5 restores >100 steps/contact and sub-0.05% error across that")
    print("range.")
    print()
    print("Default changed: PhysicsEngine.dt and the GUI's pre-filled Sim-Dt field are")
    print("now 1e-5 s (was 1e-4 s), matching this result and typical practice in")
    print("comparable explicit-integration impact-biomechanics codes (e.g. MADYMO).")
    print("This is a conservative, broadly-safe default across this model suite's full")
    print("stiffness range, not a claim of optimality for every contact -- a model-")
    print("adaptive dt (set from the steepest force_curve slope actually loaded)")
    print("remains a better medium-term solution and is noted as future work.")

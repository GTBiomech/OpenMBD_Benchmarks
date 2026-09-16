"""
Run the full OpenMBD dynamics-formulation + contact verification suite.

Usage:
    python3 run_all_benchmarks.py           # full suite, default dt = 1e-5 s
    python3 run_all_benchmarks.py --fast    # coarse dt, ~20 s, sanity check only

Requires numpy and scipy, plus the OpenMBD source modules (physics_engine.py
and its dependencies) in the same directory or on PYTHONPATH. No tkinter or
GUI dependency -- this runs the physics core headlessly.

The full suite takes roughly 10 minutes: every benchmark integrates at
dt = 1e-5 s, which is 80,000 to 100,000 steps each. Progress is printed
before each benchmark starts, so a long pause with no output is expected
and is not a hang.

--fast runs the same six benchmarks at dt = 2e-4 s. Benchmarks 1-4 still
pass there; the two contact benchmarks will report FAIL, because their
tolerances are calibrated for dt = 1e-5 s and a coarse step genuinely does
resolve a stiff contact badly. That is the convergence behaviour
benchmark_6_timestep_convergence.py quantifies, not a defect. Use --fast to
check that the suite is wired up correctly, never to judge accuracy.
"""
import sys
import time

FAST = '--fast' in sys.argv
DT = 2e-4 if FAST else 1e-5
DUR = 0.2 if FAST else None

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
except ImportError as exc:
    sys.exit(f"Missing a required package ({exc.name}). Install with:\n"
             f"    pip install numpy scipy")

try:
    import benchmark_1_freebody
    import benchmark_2_twobody
    import benchmark_3_spherical
    import benchmark_4_threebody
    import benchmark_5_bouncingball
    import benchmark_5b_damped_contact
except ImportError as exc:
    sys.exit(
        f"Could not import {exc.name!r}.\n\n"
        "This suite must sit in the same directory as the OpenMBD source\n"
        "modules it tests (physics_engine.py, physics_rigid_body.py,\n"
        "models_parser.py, models_multibody.py, models_config.py,\n"
        "physics_constraints.py, physics_utils.py, physics_joint_limits.py,\n"
        "geometry_ellipsoid.py, geometry_mesh_generator.py), together with\n"
        "the five bench_*.json model files distributed alongside it.")


LABELS = {
    '1':  'Free rigid body vs. Euler equations (translation + rotation)',
    '2':  'Two-body chain (revolute), momentum/energy conservation',
    '3':  'Spherical (ball-joint) chain, momentum/L/energy conservation',
    '4':  'Three-body chain, momentum/energy conservation',
    '5':  'Bouncing ball, lossless contact vs. closed form + force law',
    '5b': 'Bouncing ball, damped contact vs. lumped mass-spring-damper',
}

# (key, module, kwargs, rough seconds at the default dt)
PLAN = [
    ('1',  benchmark_1_freebody,        dict(duration=DUR or 1.0), 55),
    ('2',  benchmark_2_twobody,         dict(duration=DUR or 0.8), 90),
    ('3',  benchmark_3_spherical,       dict(duration=DUR or 0.8), 90),
    ('4',  benchmark_4_threebody,       dict(duration=DUR or 0.8), 150),
    ('5',  benchmark_5_bouncingball,    dict(),                    110),
    ('5b', benchmark_5b_damped_contact, dict(),                    60),
]


def main():
    print("#" * 78)
    print(f"# OpenMBD verification suite -- dt = {DT:.0e} s"
          + ("  [--fast: SANITY CHECK ONLY, see module docstring]" if FAST else ""))
    if not FAST:
        print("# Full run takes roughly 10 minutes. Long pauses are expected.")
    print("#" * 78, flush=True)

    results = {}
    t_suite = time.time()
    for key, module, kwargs, secs in PLAN:
        print(f"\n>>> Benchmark {key}: {LABELS[key]}")
        print(f"    running{'' if FAST else f' (~{secs} s)'} ...", flush=True)
        t0 = time.time()
        results[key] = module.run(dt=DT, **kwargs)
        print(f"    [{time.time() - t0:.1f} s]", flush=True)

    print()
    print("#" * 78)
    print("# SUMMARY")
    print("#" * 78)
    all_passed = True
    for key in LABELS:
        passed = results[key]['passed']
        all_passed &= passed
        print(f"  Benchmark {key:<3} {LABELS[key]:<62} {'PASS' if passed else 'FAIL'}")
    print(f"  Overall: {'ALL BENCHMARKS PASSED' if all_passed else 'FAILURES DETECTED'}"
          f"   ({time.time() - t_suite:.0f} s)")
    print("#" * 78)

    if FAST and not all_passed:
        print()
        print("NOTE: --fast uses dt = 2e-4 s. The contact benchmarks (5, 5b) are")
        print("expected to FAIL at that step size -- their tolerances are set for")
        print("dt = 1e-5 s. Re-run without --fast before drawing any conclusion.")

    print()
    print("(Run 'python3 benchmark_6_timestep_convergence.py' separately for the")
    print(" timestep-convergence sweep -- it is a study, not a pass/fail test.)")
    print("(Run 'python3 make_figures.py' to regenerate the manuscript figures.)")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

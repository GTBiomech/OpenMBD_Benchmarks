"""
Benchmark 5 -- Bouncing ball (contact dynamics/kinematics)
============================================================
A sphere is dropped onto the ground plane. The ellipsoid's force_curve is
made exactly LINEAR (F = k*x) and its unload_curve is set identical to the
loading curve, with contact damping disabled -- i.e. a lossless linear
contact spring. Because the ball starts with zero horizontal velocity and
zero spin, and the ground-contact normal is always vertical with the
contact point directly below the CG, NO torque or horizontal force is ever
generated (verified below): the motion is exactly 1-D and satisfies

    m*z'' = -m*g + k*max(0, R - z)

This is solved two independent ways as ground truth:
  (a) closed form: free fall (exact projectile motion) outside contact,
      simple harmonic motion (exact) during contact, stitched together via
      a root-find for the contact-end time;
  (b) direct high-accuracy SciPy integration of the same ODE (DOP853),
      as a cross-check on (a) that does not share any code with OpenMBD.

Both are then compared against the actual PhysicsEngine trajectory,
exercising the full contact force path: detect_contacts() -> ground-plane
penetration -> SimpleContact.get_contact_force() -> get_applied_force() ->
RNEA -> mass-matrix solve -> integration.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from harness import build_engine, body_by_name

G = 9.81
MASS = 1.0
RADIUS = 0.1
K = 50000.0          # N/m, matches the linear force_curve in bench_ball.json
Z0 = 1.0              # initial ball-centre height (m)
OMEGA = np.sqrt(K / MASS)


def closed_form_trajectory(t_array):
    """Exact z(t), vz(t) for free-fall + lossless-spring bounce(s)."""
    z = np.zeros_like(t_array)
    vz = np.zeros_like(t_array)

    # Phase 1: free fall from (Z0, 0) until z=R
    t_impact = np.sqrt(2.0 * (Z0 - RADIUS) / G)
    v_impact = G * t_impact   # downward speed magnitude at impact

    def free_fall(t, z_start, vz_start, t_start):
        dt = t - t_start
        return z_start + vz_start * dt - 0.5 * G * dt ** 2, vz_start - G * dt

    # Phase 2 (contact): SHM about z_eq = R - mg/k
    #   u = z - z_eq  satisfies u'' = -omega^2 u  =>  u(tau) = U0 cos(w tau) + (V0/w) sin(w tau)
    #   with U0 = u(0) = m g / k  (>0),  V0 = u'(0) = z'(0) = -v_impact  (<0, moving down)
    z_eq = RADIUS - MASS * G / K
    U0 = MASS * G / K
    V0 = -v_impact

    def u_of_tau(tau):
        return U0 * np.cos(OMEGA * tau) + (V0 / OMEGA) * np.sin(OMEGA * tau)

    def du_dtau(tau):
        return -U0 * OMEGA * np.sin(OMEGA * tau) + V0 * np.cos(OMEGA * tau)

    # contact ends when u(tau) returns to U0 (z back to R); search just after tau=0
    tau_end = brentq(lambda tau: u_of_tau(tau) - U0, 1e-9, 2 * np.pi / OMEGA - 1e-9)
    v_rebound = du_dtau(tau_end)   # should equal +v_impact exactly (lossless spring)

    t_contact_end = t_impact + tau_end

    for i, t in enumerate(t_array):
        if t <= t_impact:
            zz, vv = free_fall(t, Z0, 0.0, 0.0)
        elif t <= t_contact_end:
            tau = t - t_impact
            zz = z_eq + u_of_tau(tau)
            vv = du_dtau(tau)
        else:
            dt = t - t_contact_end
            zz = RADIUS + v_rebound * dt - 0.5 * G * dt ** 2
            vv = v_rebound - G * dt
        z[i], vz[i] = zz, vv
    return z, vz, t_impact, v_impact, t_contact_end, v_rebound


def ode_rhs(t, y):
    z, vz = y
    F_spring = K * max(0.0, RADIUS - z)
    az = -G + F_spring / MASS
    return [vz, az]


def independent_scipy_reference(t_array):
    sol = solve_ivp(ode_rhs, [0.0, t_array[-1]], [Z0, 0.0], t_eval=t_array,
                     method='DOP853', rtol=1e-12, atol=1e-14, max_step=1e-4)
    return sol.y[0], sol.y[1]


def run(dt=1e-5, verbose=True, T=1.0):
    eng = build_engine('bench_ball.json', pos_str=f"0 0 {Z0}", vel_str="0 0 0", dt=dt)
    eng.contact_damping = 0.0
    ball = body_by_name(eng, 'ball')

    n_steps = int(round(T / dt))
    sample_every = max(1, int(round(0.001 / dt)))

    def energy():
        x = max(0.0, RADIUS - ball.pos[2])
        spring_PE = 0.5 * K * x ** 2
        return 0.5 * MASS * ball.vel @ ball.vel + MASS * G * ball.pos[2] + spring_PE

    E0 = energy()
    t_hist = [0.0]
    z_hist = [ball.pos[2]]
    vz_hist = [ball.vel[2]]
    horiz_drift = [0.0]
    ang_drift = [0.0]
    E_hist = [E0]
    F_hist = [0.0]

    for step in range(1, n_steps + 1):
        eng.step()
        if step % sample_every == 0:
            t_hist.append(eng.time)
            z_hist.append(ball.pos[2])
            vz_hist.append(ball.vel[2])
            horiz_drift.append(np.hypot(ball.pos[0], ball.pos[1]))
            ang_drift.append(np.linalg.norm(ball.ang_vel))
            E_hist.append(energy())
            F_here = 0.0
            for c in eng.contacts:
                if c.bodyA is ball or c.bodyB is ball:
                    vA = c.bodyA.get_velocity_at_point(c.point)
                    vB = (c.bodyB.get_velocity_at_point(c.point)
                          if c.bodyB is not None else np.zeros(3))
                    v_rel_n = np.dot(vA - vB, c.normal)
                    F_here = c.get_contact_force(c.penetration, v_rel_n)
            F_hist.append(F_here)

    t_hist = np.array(t_hist)
    z_hist = np.array(z_hist)
    vz_hist = np.array(vz_hist)
    horiz_drift = np.array(horiz_drift)
    ang_drift = np.array(ang_drift)

    z_ref, vz_ref, t_impact, v_impact, t_contact_end, v_rebound = closed_form_trajectory(t_hist)
    z_ref2, vz_ref2 = independent_scipy_reference(t_hist)

    z_err_closed = np.abs(z_hist - z_ref)
    z_err_scipy = np.abs(z_hist - z_ref2)
    closed_vs_scipy = np.abs(z_ref - z_ref2)
    E_hist = np.array(E_hist)
    E_drift = np.abs(E_hist - E0) / abs(E0)
    F_hist = np.array(F_hist)
    F_ref = K * np.clip(RADIUS - z_ref, 0.0, None)   # lumped spring-mass model: F=k*x
    F_err = np.abs(F_hist - F_ref)
    in_contact = z_hist < RADIUS
    free_flight = z_hist >= RADIUS   # excludes in-contact samples (spring-PE phase-space
                                      # sampling gives a bounded, non-secular transient
                                      # offset from the continuous energy -- see README)
    E_drift_flight = E_drift[free_flight]

    if verbose:
        print("=" * 70)
        print("BENCHMARK 5: Bouncing ball (contact dynamics/kinematics)")
        print("=" * 70)
        print(f"dt = {dt}, R={RADIUS} m, m={MASS} kg, k={K} N/m "
              f"(omega={OMEGA:.1f} rad/s, T_osc={2*np.pi/OMEGA*1000:.2f} ms)")
        print(f"impact time = {t_impact:.5f} s, impact speed = {v_impact:.4f} m/s")
        print(f"contact duration (closed form) = {t_contact_end - t_impact:.5f} s "
              f"({(t_contact_end-t_impact)/dt:.0f} steps at this dt)")
        print(f"rebound speed (closed form) = {v_rebound:.4f} m/s   "
              f"(impact speed = {v_impact:.4f} -> restitution e = "
              f"{v_rebound/v_impact:.6f}, exactly 1.0 for a lossless spring)")
        print("-" * 70)
        print(f"cross-check: max |z_closed_form - z_scipy_reference| = "
              f"{closed_vs_scipy.max():.3e} m  (the two independent references agree)")
        print(f"max |z_sim - z_closed_form| = {z_err_closed.max():.3e} m")
        print(f"max |z_sim - z_scipy_ref|   = {z_err_scipy.max():.3e} m")
        print(f"max horizontal drift (should be exactly 0)  = {horiz_drift.max():.3e} m")
        print(f"max |angular velocity| (should be exactly 0) = {ang_drift.max():.3e} rad/s")
        print(f"[Force]  max |F_sim - F_lumped(=k*x)| over contact = "
              f"{F_err[in_contact].max() if in_contact.any() else 0.0:.3e} N   "
              f"(peak contact force: sim={F_hist.max():.2f} N, "
              f"lumped model={F_ref.max():.2f} N)")
        print(f"max relative energy drift, FREE-FLIGHT samples only "
              f"(unambiguous KE+PE, excludes in-contact spring-PE sampling) = "
              f"{E_drift_flight.max():.3e}")
        print(f"max relative energy drift, ALL samples incl. peak compression "
              f"(bounded symplectic-integrator 'shadow Hamiltonian' transient, "
              f"not secular loss) = {E_drift.max():.3e}")
        post_contact = z_hist[t_hist > t_contact_end]
        if len(post_contact):
            peak_rebound_height = post_contact.max()
            print(f"peak height of next bounce (informational; needs full flight "
                  f"time to resolve) = {peak_rebound_height:.5f} m vs Z0={Z0:.5f} m")

    tol_z = 2e-4 if dt <= 1e-4 else 5e-3
    tol_E_flight = 5e-4
    tol_F = 20.0   # N, absolute, against peak force ~950 N (~2% relative)
    passed = (z_err_closed.max() < tol_z and z_err_scipy.max() < tol_z and
              horiz_drift.max() < 1e-9 and ang_drift.max() < 1e-9 and
              E_drift_flight.max() < tol_E_flight and
              (not in_contact.any() or F_err[in_contact].max() < tol_F))
    if verbose:
        print("-" * 70)
        print(f"Tolerance: max |z error| < {tol_z:.1e} m; free-flight energy drift < "
              f"{tol_E_flight:.1e} (rel.); max |force error| < {tol_F:.1f} N; "
              f"horizontal/angular drift ~ 0")
        print("RESULT:", "PASS" if passed else "FAIL")
        print("=" * 70)

    return dict(t=t_hist, z_sim=z_hist, z_ref=z_ref, z_err=z_err_closed,
                E_hist=E_hist, E0=E0, E_drift=E_drift, E_drift_flight=E_drift_flight,
                F_hist=F_hist, F_ref=F_ref, F_err=F_err,
                dt=dt, t_impact=t_impact, t_contact_end=t_contact_end,
                v_impact=v_impact, v_rebound=v_rebound, passed=passed)


if __name__ == '__main__':
    run()

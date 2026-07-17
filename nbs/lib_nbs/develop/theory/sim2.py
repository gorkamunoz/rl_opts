"""
Visual searcher in 2D, v2.

FIX over sim.py: contact acquisition is now tested along the WHOLE unit step
(exact segment-to-disk geometry), not just at the step endpoints. In sim.py the
walker sampled its position only every 1.0 of path length, so with r=0.5 it
stepped over targets and the contact channel acted with r_eff ~ 0.41 instead of
0.50. sim_flight.py (blind walker) substeps at h=0.4 and already had
r_eff ~ 0.50, so the two simulators disagreed by ~18% on the size of the target
-- an asymmetry that biased the visual-vs-blind comparison toward the blind
walker. Exact segment geometry gives r_eff = r with no discretisation error at
all, and no substepping cost.

The DIFFUSIVE step length is still exactly 1 (a fresh random heading every unit
of path length), so D = d/4 = 1/4 is untouched.

Geometry per step, walker at p with fresh heading h:
  d    = vector to target centre (minimum image)
  s    = d.h                      (along-ray projection)
  perp = |d - s h|                (perpendicular distance of centre from ray)
  ray hits the disk  <=>  perp <= r  and  s > 0
  entry = s - sqrt(r^2 - perp^2)  (path length to first touch of the disk)

  VISION  : ray hits, centre within VR, centre bearing within the cone
            -> the walker commits to a straight run of up to VR and captures.
  CONTACT : ray hits with entry <= 1, regardless of cone
            -> the walker simply bumps into the disk during its unit step.

Note that a vision capture is just a licence to run straight for longer: the
dash is along the SAME heading h. Contact therefore adds exactly the events
vision misses, i.e. targets outside the cone that the unit step clips anyway.
This is the 'even a blind walker bumps into things' channel.

Cost of a capture is the path length to the disk SURFACE (entry), not to the
centre, which is the other small fix over sim.py.
"""
import numpy as np
from scipy.spatial import cKDTree


def _minimg(delta, L):
    return delta - L * np.round(delta / L)


def run(L, Nt, r, VR, tau, theta, W, seed, rule, max_steps=200000):
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0, L, size=(Nt, 2))
    tree = cKDTree(targets, boxsize=L)
    ti = rng.integers(0, Nt, size=W)
    a0 = rng.uniform(0, 2 * np.pi, size=W)
    pos = (targets[ti] + tau * np.column_stack([np.cos(a0), np.sin(a0)])) % L
    path = np.zeros(W)
    done = np.zeros(W, bool)
    idx = np.arange(W)
    half = theta / 2.0
    Rsearch = max(VR, 1.0 + r) + 1e-9   # covers vision (VR) and contact (step+r)

    for _ in range(max_steps):
        act = ~done
        if not act.any():
            break
        p = pos[act]
        ai = idx[act]

        if rule == 'iso':
            # perfect-reaim baseline: absorbing disk of radius VR, r plays no role
            dist, _ = tree.query(p, k=1)
            hit = dist <= VR
            if hit.any():
                gi = ai[hit]
                path[gi] += dist[hit]
                done[gi] = True
            mv = ~hit
            if mv.any():
                ang = rng.uniform(0, 2 * np.pi, size=int(mv.sum()))
                pos[ai[mv]] = (p[mv] + np.column_stack([np.cos(ang), np.sin(ang)])) % L
                path[ai[mv]] += 1.0
            continue

        # ---- directional walker: vision channel + contact channel ----
        ang = rng.uniform(0, 2 * np.pi, size=len(p))
        h = np.column_stack([np.cos(ang), np.sin(ang)])
        capt = np.zeros(len(p), bool)
        cost = np.zeros(len(p))
        neigh = tree.query_ball_point(p, Rsearch)

        for k in range(len(p)):
            lst = neigh[k]
            if not lst:
                continue
            d = _minimg(targets[lst] - p[k], L)
            dd = np.hypot(d[:, 0], d[:, 1])
            s = d[:, 0] * h[k, 0] + d[:, 1] * h[k, 1]          # along-ray
            cross = h[k, 0] * d[:, 1] - h[k, 1] * d[:, 0]
            perp = np.abs(cross)                                # |d - s h|
            ray_hits = (perp <= r) & (s > 0)
            if not ray_hits.any():
                continue
            # path length to first touch of the disk surface
            entry = np.where(ray_hits, s - np.sqrt(np.clip(r**2 - perp**2, 0, None)), np.inf)
            entry = np.clip(entry, 0.0, None)
            bear = np.abs(np.arctan2(cross, s))                 # bearing of centre
            vision = ray_hits & (dd <= VR) & (bear <= half)
            contact = ray_hits & (entry <= 1.0)
            ok = vision | contact
            if ok.any():
                capt[k] = True
                cost[k] = entry[ok].min()

        if capt.any():
            gi = ai[capt]
            path[gi] += cost[capt]
            done[gi] = True
        mv = ~capt
        if mv.any():
            pos[ai[mv]] = (p[mv] + h[mv]) % L
            path[ai[mv]] += 1.0

    return path, done


def mfpt(**kw):
    path, done = run(**kw)
    frac = 1 - done.mean()
    p = path[done]
    return p.mean(), p.std() / np.sqrt(max(len(p), 1)), frac


if __name__ == "__main__":
    import time
    L, Nt, r = 300, 90, 0.5
    rho = Nt / L**2
    A = 2 / (np.pi * rho)
    t0 = time.time()
    # blind-mode test: theta->0 kills vision, leaving only contact.
    # Should give T = A ln(tau/r_eff) with r_eff = r = 0.5 (no discretisation loss).
    print("blind-mode check (theta->0): r_eff should now be ~%.2f" % r)
    for tv in [4.0, 6.0, 10.0]:
        m, s, f = mfpt(L=L, Nt=Nt, r=r, VR=3.0, tau=tv, theta=0.02,
                       W=2000, seed=5, rule='dir')
        reff = tv * np.exp(-m / A)
        print("  tau=%4.1f  T=%7.1f  r_eff=%.3f  [%.0fs]" % (tv, m, reff, time.time() - t0))

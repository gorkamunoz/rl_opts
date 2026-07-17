import numpy as np
from scipy.spatial import cKDTree

# ---------- model ----------
# 2D box L x L, periodic. Nt targets radius r. Walker unit steps (d=1), fresh
# random direction each step  ->  D = d/4 = 1/4, time = path length = #steps.
# rule 'iso' : capture when nearest target center within VR (perfect reaim).
# rule 'dir' : capture only if a target center is within VR AND within forward
#              cone (half angle theta/2 of heading) AND heading ray passes
#              within r of the center (a "hit"); otherwise a miss -> keep BM.
# start: distance tau from a random (live) target, random heading.

def _minimg(delta, L):
    return delta - L*np.round(delta/L)

def run(L, Nt, r, VR, tau, theta, W, seed, rule, max_steps=200000):
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0, L, size=(Nt, 2))
    tree = cKDTree(targets, boxsize=L)
    ti = rng.integers(0, Nt, size=W)
    a0 = rng.uniform(0, 2*np.pi, size=W)
    pos = (targets[ti] + tau*np.column_stack([np.cos(a0), np.sin(a0)])) % L
    path = np.zeros(W)
    done = np.zeros(W, bool)
    idx = np.arange(W)
    half = theta/2.0

    for step in range(max_steps):
        act = ~done
        if not act.any():
            break
        p = pos[act]
        ai = idx[act]

        if rule == 'iso':
            dist, j = tree.query(p, k=1)
            hit = dist <= VR
            if hit.any():
                gi = ai[hit]
                path[gi] += dist[hit]      # ballistic dash to target
                done[gi] = True
            mv = ~hit
            if mv.any():
                ang = rng.uniform(0, 2*np.pi, size=mv.sum())
                pos[ai[mv]] = (p[mv] + np.column_stack([np.cos(ang), np.sin(ang)])) % L
                path[ai[mv]] += 1.0
        else:  # 'dir' : two channels -- (1) contact within r, (2) cone detect+dash
            ang = rng.uniform(0, 2*np.pi, size=len(p))
            h = np.column_stack([np.cos(ang), np.sin(ang)])
            d1, _ = tree.query(p, k=1)             # nearest center
            capt = d1 <= r                         # channel 1: physical contact
            dash = np.where(capt, d1, 0.0)
            neigh = tree.query_ball_point(p, VR)   # candidates within VR (detection)
            nz = [k for k in range(len(p)) if (not capt[k]) and neigh[k]]
            for k in nz:
                lst = neigh[k]
                d = _minimg(targets[lst] - p[k], L)      # vectors to targets
                dd = np.hypot(d[:, 0], d[:, 1])
                # bearing relative to heading
                cross = h[k, 0]*d[:, 1] - h[k, 1]*d[:, 0]
                dot = h[k, 0]*d[:, 0] + h[k, 1]*d[:, 1]
                bear = np.abs(np.arctan2(cross, dot))     # in [0,pi]
                incone = bear <= half
                perp = dd*np.sin(bear)
                hitc = (perp <= r) & (dot > 0)
                ok = incone & hitc & (dd <= VR)
                if ok.any():
                    capt[k] = True
                    dash[k] = dd[ok].min()
            if capt.any():
                gi = ai[capt]
                path[gi] += dash[capt]
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
    return p.mean(), p.std()/np.sqrt(len(p)), frac

if __name__ == "__main__":
    import json, sys
    print(mfpt(L=200, Nt=40, r=0.5, VR=2.0, tau=5.0, theta=2*np.pi, W=2000, seed=1, rule='iso'))

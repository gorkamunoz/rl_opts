import numpy as np
from scipy.spatial import cKDTree

# Blind ballistic-flight searcher. Step lengths drawn from a mixture of
# exponentials (bi-exponential a la Ferreira). No vision: a target is acquired
# only on physical contact, i.e. when the flight path comes within r of a center.
# Time = path length (unit speed). Start at distance tau from a random target.

def run_flight(L, Nt, r, tau, dscales, weights, W, seed, h=0.25, max_path=40000):
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0, L, size=(Nt, 2))
    tree = cKDTree(targets, boxsize=L)
    ti = rng.integers(0, Nt, size=W)
    a0 = rng.uniform(0, 2*np.pi, size=W)
    pos = (targets[ti] + tau*np.column_stack([np.cos(a0), np.sin(a0)])) % L
    head = np.zeros((W, 2))
    remaining = np.zeros(W)
    path = np.zeros(W)
    done = np.zeros(W, bool)
    dscales = np.asarray(dscales, float)
    cw = np.cumsum(np.asarray(weights, float)); cw /= cw[-1]

    def new_flights(mask):
        n = int(mask.sum())
        if n == 0:
            return
        u = rng.random(n)
        mode = np.searchsorted(cw, u)
        Ls = -dscales[mode]*np.log(rng.random(n))
        ang = rng.uniform(0, 2*np.pi, n)
        head[mask] = np.column_stack([np.cos(ang), np.sin(ang)])
        remaining[mask] = Ls

    new_flights(~done)
    nsteps = int(max_path/h)
    for _ in range(nsteps):
        act = ~done
        if not act.any():
            break
        need = act & (remaining <= 0)
        new_flights(need)
        # advance active
        pos[act] = (pos[act] + h*head[act]) % L
        remaining[act] -= h
        path[act] += h
        d1, _ = tree.query(pos[act], k=1)
        hit = d1 <= r
        if hit.any():
            gi = np.where(act)[0][hit]
            done[gi] = True
    p = path[done]
    return p.mean(), p.std()/np.sqrt(max(len(p),1)), 1-done.mean()

if __name__ == "__main__":
    import time
    t=time.time()
    # ballistic ceiling check: single huge scale -> T ~ 1/(2 r rho)
    L,Nt,r=300,90,0.5; rho=Nt/L**2
    m,s,f=run_flight(L,Nt,r,tau=6.0,dscales=[1e5],weights=[1.0],W=3000,seed=1)
    print("ballistic T=%.1f SEM=%.1f cens=%.3f | mean-free-path 1/(2 r rho)=%.0f | %.0fs"
          %(m,s,f,1/(2*r*rho),time.time()-t))

import numpy as np, json, os, time, sys
from sim2 import mfpt
L, Nt, r = 300, 90, 0.5
taus = [2, 4, 6, 8, 10, 12, 16, 20, 25, 30]
VRs = [1, 2, 3, 5, 8, 12]
W = 1000
FN = '/home/claude/visual_cache_v2.json'
cache = json.load(open(FN)) if os.path.exists(FN) else {}
t0 = time.time()
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 260
for tv in taus:
    for vr in VRs:
        k = '%d,%d' % (tv, vr)
        if k in cache:
            continue
        if time.time() - t0 > budget:
            print('BUDGET; %d/%d done' % (len(cache), len(taus)*len(VRs))); json.dump(cache, open(FN,'w')); sys.exit(0)
        m, s, f = mfpt(L=L, Nt=Nt, r=r, VR=float(vr), tau=float(tv),
                       theta=2*np.pi, W=W, seed=300+tv*7+vr, rule='dir')
        cache[k] = m
        print('tau=%2d VR=%2d T=%7.1f [%.0fs]' % (tv, vr, m, time.time()-t0), flush=True)
    json.dump(cache, open(FN, 'w'))
json.dump(cache, open(FN, 'w'))
print('GRID COMPLETE %d cells [%.0fs]' % (len(cache), time.time()-t0))

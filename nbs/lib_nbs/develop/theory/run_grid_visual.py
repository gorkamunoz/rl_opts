import numpy as np, json, time, os
from sim import mfpt
L,Nt,r=300,90,0.5
taus=[2,4,6,8,10,12]
VRs=[1,2,3,5,8,12]
W=1200
t0=time.time()
grid=np.full((len(taus),len(VRs)),np.nan)
fn='/home/claude/visual_grid.json'
if os.path.exists(fn):
    grid=np.array(json.load(open(fn))['T'])
for i,tv in enumerate(taus):
    for j,vr in enumerate(VRs):
        if not np.isnan(grid[i,j]): continue
        m,s,f=mfpt(L=L,Nt=Nt,r=r,VR=float(vr),tau=float(tv),theta=2*np.pi,W=W,seed=100+i*10+j,rule='dir')
        grid[i,j]=m
        print("tau=%2d VR=%2d  T=%7.1f  [%.0fs]"%(tv,vr,m,time.time()-t0),flush=True)
    json.dump({'taus':taus,'VRs':VRs,'T':grid.tolist()},open(fn,'w'))
print("VISUAL GRID DONE [%.0fs]"%(time.time()-t0))

import numpy as np, json, time
from sim_flight import run_flight
from sim import mfpt as visual_mfpt

t0=time.time(); L,Nt,r=300,90,0.5; rho=Nt/L**2
A=2/(np.pi*rho); VR=3.0; lam=1/(2*r*rho); lt=1/np.sqrt(rho)
print("A=%.0f  mean-free-path=%.0f  inter-target=%.1f"%(A,lam,lt))
def el(): return time.time()-t0

taus=[3,6,9,12]
res={'A':A,'VR':VR,'lam':lam,'r':r,'rho':rho,'taus':taus}

# --- blind bi-exp: optimize long-mode weight at each tau (d2=tau intensive, d1=1000 relocate) ---
d1=1000.0
wl_grid=[0.0,0.1,1.0]
biexp=[]
for tv in taus:
    best=None
    for wl in wl_grid:
        ds=[float(tv), d1]; ws=[1-wl, wl]
        m,s,f=run_flight(L,Nt,r,float(tv),ds,ws,W=1200,seed=7,h=0.4)
        if best is None or m<best[1]:
            best=(wl,m,s)
    wl=best[0]                       # rerun best with more walkers
    m,s,f=run_flight(L,Nt,r,float(tv),[float(tv),d1],[1-wl,wl],W=5000,seed=8,h=0.4)
    biexp.append((tv,m,s,wl))
    print("BIEXP tau=%2d  T=%7.1f SEM=%5.1f  (opt w_long=%.2f)  [%.0fs]"%(tv,m,s,wl,el()),flush=True)
res['biexp']=biexp

# --- single exponential scale=tau (pure intensive, w_long=0) reference ---
singexp=[]
for tv in taus:
    m,s,f=run_flight(L,Nt,r,float(tv),[float(tv)],[1.0],W=5000,seed=9,h=0.4)
    singexp.append((tv,m,s)); print("SINGLE tau=%2d T=%7.1f [%.0fs]"%(tv,m,el()),flush=True)
res['singexp']=singexp

# --- visual walker (two-channel: cone theta=2pi, VR=3, contact r) ---
vis=[]
for tv in taus:
    m,s,f=visual_mfpt(L=L,Nt=Nt,r=r,VR=VR,tau=float(tv),theta=2*np.pi,W=5000,seed=10,rule='dir')
    vis.append((tv,m,s)); print("VISUAL tau=%2d T=%7.1f [%.0fs]"%(tv,m,el()),flush=True)
res['visual']=vis

json.dump(res,open('/home/claude/results3.json','w'),indent=1)
print("done [%.0fs]"%el())

import numpy as np, json, time
from sim import mfpt

t0 = time.time()
res = {}
L, Nt = 300, 90
rho = Nt/L**2
A = 2/(np.pi*rho)
VR = 3.0
res['A_pred'] = A; res['VR'] = VR

def log(msg): print("%s  [%.0fs]" % (msg, time.time()-t0))

# 1) iso baseline tau-scan (absorbing at VR, unchanged)
tau_scan = []
for tv in [4,5,6,8,10]:
    m,s,f = mfpt(L=L,Nt=Nt,r=0.5,VR=VR,tau=float(tv),theta=2*np.pi,W=8000,seed=11,rule='iso')
    tau_scan.append((tv,m,s,f)); log("tau=%2d T=%7.1f"%(tv,m))
res['tau_scan']=tau_scan

# 2) iso baseline rho-scan
rho_scan=[]
for nt in [45,90,180,360]:
    m,s,f = mfpt(L=L,Nt=nt,r=0.5,VR=2.0,tau=5.0,theta=2*np.pi,W=8000,seed=22,rule='iso')
    rho_scan.append((nt/L**2,m,s,f)); log("rho=%.4g T=%7.1f"%(nt/L**2,m))
res['rho_scan']=rho_scan; res['tau2']=5.0; res['VR2']=2.0

# iso reference at tau=6, VR=3
tauc=6.0
m_iso,s_iso,f=mfpt(L=L,Nt=Nt,r=0.5,VR=VR,tau=tauc,theta=2*np.pi,W=8000,seed=33,rule='iso')
res['iso_ref']=(m_iso,s_iso); log("ISO ref T=%.1f"%m_iso)

# 3) BLIND floor: dir with tiny theta -> only contact channel (absorbing radius r)
blind=[]
for r in [0.5,1.0,1.5,2.0]:
    m,s,f=mfpt(L=L,Nt=Nt,r=r,VR=VR,tau=tauc,theta=0.02,W=6000,seed=66,rule='dir')
    blind.append((r,m,s,f)); log("BLIND r=%.1f T=%7.1f (pred A ln(tau/r)=%.0f)"%(r,m,A*np.log(tauc/r)))
res['blind']=blind

# 4) directional collapse (both channels active)
cone=[]
for r in [0.5,1.0,2.0]:                        # r sweep, wide cone
    m,s,f=mfpt(L=L,Nt=Nt,r=r,VR=VR,tau=tauc,theta=2*np.pi,W=6000,seed=44,rule='dir')
    cone.append(('r',r,2*np.pi,m,s,f)); log("r=%.1f th=2pi T=%7.1f"%(r,m))
for th in [0.2,0.4,0.6,0.8,1.5,2*np.pi]:       # theta sweep, r=1.5
    m,s,f=mfpt(L=L,Nt=Nt,r=1.5,VR=VR,tau=tauc,theta=float(th),W=6000,seed=55,rule='dir')
    cone.append(('theta',th,1.5,m,s,f)); log("th=%.2f r=1.5 T=%7.1f"%(th,m))
res['cone']=cone

json.dump(res,open('/home/claude/results2.json','w'),indent=1)
log("done")

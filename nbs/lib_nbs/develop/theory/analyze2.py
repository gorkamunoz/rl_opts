import numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('/home/claude/results2.json'))
A, VR = R['A_pred'], R['VR']
tauc = 6.0

# baseline
ts=np.array(R['tau_scan']); lt=np.log(ts[:,0]); m=ts[:,0]<=6
Afit,Bfit=np.polyfit(lt[m],ts[m,1],1)
rs=np.array(R['rho_scan']); inv=1/rs[:,0]; sl,ic=np.polyfit(inv,rs[:,1],1)
print("baseline: ln-tau slope %.0f (pred %.0f);  T~1/rho, T*rho=%s"%(Afit,A,np.round(rs[:,1]*rs[:,0],3)))

m_iso,s_iso=R['iso_ref']
# effective iso absorbing radius from iso ref
VReff_iso = tauc*np.exp(-(m_iso-(VR**2-tauc**2))/A)
print("VReff_iso=%.2f (VR=%.1f)"%(VReff_iso,VR))

# blind -> discrete effective contact radius r_eff(r)
bl=np.array(R['blind'])   # r,T,SEM,cens
r_nom=bl[:,0]; Tbl=bl[:,1]
reff = tauc*np.exp(-(Tbl-(bl[:,0]**2-tauc**2))/A)   # invert with dash~r term
print("blind: r_nom=%s -> r_eff=%s"%(r_nom, np.round(reff,2)))
reff_of = dict(zip(np.round(r_nom,2), reff))

def pc_exact(theta,r): return min(theta,2*np.arcsin(min(r/VR,1)))/(2*np.pi)

# directional -> VReff_dir
rows=[]
for kind,val,other,Td,sd,f in R['cone']:
    if kind=='r': r,theta=val,other
    else: theta,r=val,other
    pc=pc_exact(theta,r)
    VReff=VReff_iso*np.exp(-(Td-m_iso)/A)
    rc=reff_of.get(round(r,2), r)          # discrete contact radius for this r
    rows.append((kind,r,theta,pc,VReff,rc,Td,sd))
    print("%-5s r=%.1f th=%.2f pc=%.3f  VReff_dir=%.2f  (vision=%.2f, contact r_eff=%.2f)"
          %(kind,r,theta,pc,VReff, VR*pc**0.36, rc))

# fit beta on vision-dominated points (VReff clearly above contact floor)
arr=np.array([(x[3],x[4],x[5]) for x in rows])   # pc, VReff, rc
pc=arr[:,0]; VReff=arr[:,1]; rc=arr[:,2]
vis2 = np.clip(VReff**2 - rc**2, 1e-6, None)      # subtract contact floor in quadrature
y=0.5*np.log(vis2); x=np.log(pc)                  # ln(vision) = ln VR + beta ln pc
b_beta,b_int=np.polyfit(x,y,1)
print("\nquadrature model VReff^2 = r_eff^2 + (VR' pc^beta)^2 :  beta=%.2f  VR'=%.2f"
      %(b_beta,np.exp(b_int)))

# ---- figure ----
fig,ax=plt.subplots(2,2,figsize=(11,8))
# a
ax[0,0].errorbar(lt,ts[:,1],yerr=ts[:,2],fmt='o'); xx=np.linspace(lt.min(),lt.max(),20)
ax[0,0].plot(xx,Afit*xx+Bfit,'--',label='slope %.0f (pred %.0f)'%(Afit,A))
ax[0,0].set_xlabel('ln tau'); ax[0,0].set_ylabel('T'); ax[0,0].set_title('(a) baseline T vs ln tau'); ax[0,0].legend(fontsize=8)
# b
ax[0,1].errorbar(inv,rs[:,1],yerr=rs[:,2],fmt='s',color='C1'); xx=np.linspace(0,inv.max()*1.05,20)
ax[0,1].plot(xx,sl*xx+ic,'--',color='C1',label='slope %.3f'%sl)
ax[0,1].set_xlabel('1/rho'); ax[0,1].set_ylabel('T'); ax[0,1].set_title('(b) baseline T ~ 1/rho'); ax[0,1].legend(fontsize=8)
# c  blind floor
ax[1,0].errorbar(np.log(tauc/r_nom),Tbl,yerr=bl[:,2],fmt='D',color='C4',label='blind sim')
xx=np.linspace(0.9,2.6,20); ax[1,0].plot(xx,A*xx,'k--',label='A ln(tau/r) continuum')
ax[1,0].set_xlabel('ln(tau/r)'); ax[1,0].set_ylabel('T (blind, theta->0)')
ax[1,0].set_title('(c) NEW: blind searcher finite, absorb radius r'); ax[1,0].legend(fontsize=8)
# d  two-channel effective radius
mr=np.array([x[0]=='r' for x in rows])
ax[1,1].errorbar(pc[mr],VReff[mr],fmt='o',color='C3',label='r sweep (theta=2pi)')
ax[1,1].errorbar(pc[~mr],VReff[~mr],fmt='^',color='C2',label='theta sweep (r=1.5)')
pp=np.linspace(0.02,0.6,50)
ax[1,1].plot(pp,VR*pp**0.36,'k:',label='vision  VR pc^0.36')
ax[1,1].axhline(reff_of[1.5],color='C2',ls='--',lw=1,label='contact floor r_eff(1.5)')
ax[1,1].set_xscale('log'); ax[1,1].set_xlabel('p_c'); ax[1,1].set_ylabel('VReff_dir')
ax[1,1].set_title('(d) VReff interpolates vision (VR) <-> contact (r)'); ax[1,1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('/home/claude/mfpt_tests2.png',dpi=130)
print("saved")

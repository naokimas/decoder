# =====================================================================
# SI Section S10: Robustness of the cross-transfer network structure
#
# This script: Table S12, robustness to the two edge thresholds
# Produces   : tab_thresh.tex
# Runtime    : seconds
# See ../README.md for the run order and for where the
# equilibrium data comes from.
# =====================================================================
"""How far the two edge thresholds can move before the component partition changes."""
import numpy as np, networkx as nx
from pathlib import Path
HERE=Path(__file__).resolve().parent
d=np.load(HERE.parent/"shared_data"/"fig45_matrices.npz",allow_pickle=True)
LAB=[str(x) for x in d["dynamics"]]; n=len(LAB)
PAN=[("NN, MinCorr","nn","minc"),("NN, QR","nn","qr"),("NN, random","nn","random"),
     ("linear, MinCorr","lin","minc"),("linear, QR","lin","qr"),
     ("linear, random","lin","random")]

def part(dec,sent,nt,rt):
    M=d[f"M_{dec}_{sent}"]; B=d[f"B_{dec}_{sent}"]
    with np.errstate(divide="ignore",invalid="ignore"):
        R=np.where(np.isfinite(B)&(B>0),M/B,np.nan)
    G=nx.DiGraph(); G.add_nodes_from(LAB)
    for i in range(n):
        for j in range(n):
            if i!=j and M[i,j]<nt and R[i,j]<rt: G.add_edge(LAB[i],LAB[j])
    return frozenset(frozenset(c) for c in nx.weakly_connected_components(G))

rows=[]
print(f"  {'panel':16s} {'ref sizes':>14s} {'min r':>8s} {'min nRMSE':>10s} {'min nRMSE':>10s}")
print(f"  {'':16s} {'(0.15, 0.8)':>14s} {'@0.15':>8s} {'@r=0.8':>10s} {'@r=0.65':>10s}")
for title,dec,sent in PAN:
    ref=part(dec,sent,0.15,0.80)
    sizes=sorted((len(c) for c in ref),reverse=True)
    # lower r with n-RMSE fixed at 0.15
    lo=0.80
    for rt in np.arange(0.80,0.0,-0.001):
        if part(dec,sent,0.15,rt)!=ref: break
        lo=rt
    # lower n-RMSE with r fixed
    mins=[]
    for rt in (0.80,0.65):
        refr=part(dec,sent,0.15,rt) if rt!=0.80 else ref
        m=0.15
        for nt in np.arange(0.15,0.0,-0.001):
            if part(dec,sent,nt,rt)!=refr: break
            m=nt
        mins.append(m)
    # raising the n-RMSE threshold
    same_up=all(part(dec,sent,nt,0.80)==ref for nt in (0.2,0.3,0.5,1.0,5.0))
    rows.append((title,sizes,lo,mins[0],mins[1],same_up))
    print(f"  {title:16s} {str(sizes):>14s} {lo:8.3f} {mins[0]:10.3f} {mins[1]:10.3f}"
          f"   raising n-RMSE keeps partition: {same_up}")
body=[]
for title,sizes,lo,m1,m2,_ in rows:
    body.append(f"{title} & {lo:.3f} & {m1:.3f} & {m2:.3f} \\\\")
body[-1]=body[-1].rstrip()[:-2].rstrip()
(HERE/"tables").mkdir(exist_ok=True)
(HERE/"tables"/"tab_thresh.tex").write_text("\n".join(body)+"\n")
print("\n  wrote tab_thresh.tex")

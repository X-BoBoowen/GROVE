import numpy as np
rng=np.random.default_rng(5)
def trf(X,kind,p):
    lam=np.linalg.eigvals(X)
    v = lam**(-p) if kind=="pow" else np.exp(-p*lam)
    return v.sum().real, abs(v.sum().imag)
worst={}; maxim=0
for trial in range(4000):
    n=int(rng.integers(4,9)); W=(rng.random((n,n))<0.35)*rng.lognormal(0,1,(n,n)); np.fill_diagonal(W,0)
    for i in range(n): W[i,(i+1)%n]=max(W[i,(i+1)%n],0.5)   # directed cycle -> strongly connected
    M=np.diag(W.sum(1))-W + 0.05*np.eye(n)                   # directed (out-degree) Laplacian + killing, NOT symmetric
    k=int(rng.integers(0,n-3)); S=list(rng.choice(n,k,replace=False)) if k else []
    rest=[v for v in range(n) if v not in S]; a,b=rng.choice(rest,2,replace=False)
    for kind,p in [("pow",0.5),("pow",1.0),("pow",2.0),("heat",1.0)]:
        def H(T):
            keep=[v for v in range(n) if v not in T]; val,im=trf(M[np.ix_(keep,keep)],kind,p); return val,im
        vals=[H(S),H(S+[a]),H(S+[b]),H(S+[a,b])]; maxim=max(maxim,max(x[1] for x in vals))
        v=[x[0] for x in vals]; scale=abs(v[0]-v[1])+abs(v[0]-v[2])+1e-300
        key=f"{kind} {p}"; worst[key]=min(worst.get(key,np.inf),(v[3]-v[1]-v[2]+v[0])/scale)
        mono=min(v[0]-v[1],v[0]-v[2]); worst[key+" mono"]=min(worst.get(key+" mono",np.inf),mono/scale)
print("max |imag part of trace| =",maxim)
for k,w in worst.items(): print(f"directed hard grounding {k:16s} min rel = {w:+.2e}")

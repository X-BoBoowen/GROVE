import numpy as np, itertools
rng=np.random.default_rng(11)
def rand_graph():
    n=int(rng.integers(5,11)); W=np.zeros((n,n)); p=rng.permutation(n)
    for a,b in zip(p[:-1],p[1:]): W[a,b]=W[b,a]=rng.lognormal(0,1)
    for i,j in itertools.combinations(range(n),2):
        if W[i,j]==0 and rng.random()<0.25: W[i,j]=W[j,i]=rng.lognormal(0,1)
    return n,np.diag(W.sum(1))-W
def trf(X,f): return f(np.linalg.eigvalsh(X)).sum()
def test(f,mode,trials=2500):
    worst=np.inf
    for _ in range(trials):
        n,L=rand_graph(); M=L+0.01*np.eye(n)
        k=int(rng.integers(0,n-3)); S=list(rng.choice(n,k,replace=False)) if k else []
        rest=[v for v in range(n) if v not in S]; a,b=rng.choice(rest,2,replace=False)
        def H(T):
            if mode=="hard":
                keep=[v for v in range(n) if v not in T]; return trf(M[np.ix_(keep,keep)],f)
            X=M.copy()
            for v in T: X[v,v]+=2.0
            return trf(X,f)
        v=[H(S),H(S+[a]),H(S+[b]),H(S+[a,b])]
        scale=abs(v[0]-v[1])+abs(v[0]-v[2])+1e-300
        worst=min(worst,(v[3]-v[1]-v[2]+v[0])/scale)
    return worst
fs={"heat exp(-0.5x)":lambda x:np.exp(-0.5*x),"heat exp(-5x)":lambda x:np.exp(-5*x),
    "x^-5":lambda x:x**-5.0,"1/(x+1)^2":lambda x:(x+1)**-2.0,
    "NOT CM: x^-0.5*cos(x) control":lambda x:np.cos(x)/np.sqrt(x)}
for name,f in fs.items():
    print(f"{name:32s} hard min rel={test(f,'hard'):+.2e}   soft min rel={test(f,'soft'):+.2e}")

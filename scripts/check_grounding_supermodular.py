import numpy as np, itertools
rng=np.random.default_rng(7)
def rand_graph(weighted):
    n=int(rng.integers(5,11))
    W=np.zeros((n,n))
    perm=rng.permutation(n)
    for a,b in zip(perm[:-1],perm[1:]):  # random spanning path -> connected
        W[a,b]=W[b,a]=rng.lognormal(0,1) if weighted else 1
    for i,j in itertools.combinations(range(n),2):
        if W[i,j]==0 and rng.random()<0.25: W[i,j]=W[j,i]=rng.lognormal(0,1) if weighted else 1
    return n,np.diag(W.sum(1))-W
def mpow(M,p):
    w,U=np.linalg.eigh(M); return w**p
def tr_pow(M,p): return (np.linalg.eigvalsh(M)**p).sum()
# V3 hard grounding, negative power: H(S)=tr(((L+sigma I)_{-S})^{-s}); want decreasing+supermodular: H(S+a+b)-H(S+a)-H(S+b)+H(S) >= 0
def H_hard(L,S,s,sig):
    keep=[v for v in range(L.shape[0]) if v not in S]
    if not keep: return 0.0
    return tr_pow(L[np.ix_(keep,keep)]+sig*np.eye(len(keep)),-s)
# V4 soft grounding, negative power: tr((L+sigma I+beta sum e e^T)^{-s})
def H_soft(L,S,s,sig,beta):
    M=L+sig*np.eye(L.shape[0]); 
    for v in S: M[v,v]+=beta
    return tr_pow(M,-s)
# V1 soft grounding, positive power gain: F(S)=tr((I+alpha L+beta sum e e^T)^s) ; want submodular: mixed <= 0
def F_soft(L,S,s,alpha,beta):
    M=np.eye(L.shape[0])+alpha*L
    for v in S: M[v,v]+=beta
    return tr_pow(M,s)
def test(fn, sign, trials=4000):
    worst=np.inf; rel=np.inf
    for _ in range(trials):
        weighted=rng.random()<0.5
        n,L=rand_graph(weighted)
        k=int(rng.integers(0,n-3))
        S=list(rng.choice(n,k,replace=False)) if k else []
        rest=[v for v in range(n) if v not in S]
        a,b=rng.choice(rest,2,replace=False)
        try: vals=[fn(L,S),fn(L,S+[a]),fn(L,S+[b]),fn(L,S+[a,b])]
        except np.linalg.LinAlgError: continue
        if not np.all(np.isfinite(vals)): continue
        mixed=sign*(vals[3]-vals[1]-vals[2]+vals[0])
        scale=abs(vals[0]-vals[1])+abs(vals[0]-vals[2])+1e-300
        worst=min(worst,mixed); rel=min(rel,mixed/scale)
    return worst,rel
print("V3 hard grounding tr(((L+sI)_{-S})^{-s})  [supermodular needs >=0]")
for sig in [0.0,0.01,1.0]:
    for s in [0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0]:
        fn=lambda L,S,s=s,sig=sig: H_hard(L, S if (S or sig>0) else S, s, sig)
        # sigma=0 needs S nonempty: enforce by always grounding vertex 0 as seed
        if sig==0.0:
            fn=lambda L,S,s=s: H_hard(L, list(set(S)|{0}) if 0 not in S else S, s, 0.0)
        w,r=test(fn,+1,2500); print(f"  sigma={sig:<5} s={s:<4} min mixed={w:+.2e}  min rel={r:+.2e}")
print("V4 soft grounding tr((L+sI+beta*E_S)^{-s})  [supermodular needs >=0]")
for beta in [0.1,1.0,10.0,1e4]:
    for s in [0.25,0.5,1.0]:
        w,r=test(lambda L,S,s=s,beta=beta:H_soft(L,S,s,0.01,beta),+1,2500); print(f"  beta={beta:<7} s={s:<4} min mixed={w:+.2e}  min rel={r:+.2e}")
print("V1 soft grounding positive power tr((I+aL+beta*E_S)^s)  [submodular needs <=0, shown as -mixed>=0]")
for beta in [0.1,10.0]:
    for s in [0.25,0.5,0.85]:
        w,r=test(lambda L,S,s=s,beta=beta:F_soft(L,S,s,1.0,beta),-1,2500); print(f"  beta={beta:<5} s={s:<4} min(-mixed)={w:+.2e} min rel={r:+.2e}")

import numpy as np, itertools
rng=np.random.default_rng(1)
def lap(n,E):
    A=np.zeros((n,n))
    for u,v in E: A[u,v]=A[v,u]=1
    return np.diag(A.sum(1))-A
def edge_test(p, sigma, trials=3000):
    worst=0;ex=None
    for _ in range(trials):
        n=rng.integers(4,8)
        E=[(i,(i+1)%n) for i in range(n-1)]  # path, connected
        E+= [e for e in itertools.combinations(range(n),2) if rng.random()<0.2 and e not in E]
        non=[e for e in itertools.combinations(range(n),2) if e not in E and (e[1],e[0]) not in E]
        if len(non)<2: continue
        i,j=rng.choice(len(non),2,replace=False); e,f=non[i],non[j]
        def H(EE):
            w=np.linalg.eigvalsh(lap(n,EE)+sigma*np.eye(n)); return (w**(-p)).sum()
        mixed=H(E+[e,f])-H(E+[e])-H(E+[f])+H(E)   # supermodular (for decreasing fn) needs >=0
        if mixed<worst: worst=mixed; ex=(n,E,e,f)
    return worst,ex
def ground_test(p, trials=3000):
    worst=0
    for _ in range(trials):
        n=rng.integers(5,9)
        E=[(i,i+1) for i in range(n-1)]+[e for e in itertools.combinations(range(n),2) if rng.random()<0.25]
        L=lap(n,list(set(E)))
        a,b=rng.choice(n,2,replace=False); base=[int(rng.integers(n))]
        if a in base or b in base: continue
        def H(S):
            keep=[v for v in range(n) if v not in S]; w=np.linalg.eigvalsh(L[np.ix_(keep,keep)]); return (w**(-p)).sum()
        mixed=H(base+[a,b])-H(base+[a])-H(base+[b])+H(base)
        worst=min(worst,mixed)
    return worst
for p in [1,2]:
    for sg in [0.01,1.0]:
        w,ex=edge_test(p,sg); print(f"edge add, power -{p}, sigma={sg}: min mixed diff = {w:.3e}", "" if ex is None else f"e.g. n={ex[0]} e={ex[2]} f={ex[3]}")
for p in [1,2]:
    print(f"node grounding tr(L_-S^-{p}): min mixed diff = {ground_test(p):.3e}")

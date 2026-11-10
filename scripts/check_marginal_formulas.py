import numpy as np, math
from scipy import integrate
rng=np.random.default_rng(3); n=9
W=np.triu((rng.random((n,n))<0.4)*rng.lognormal(0,1,(n,n)),1); W=W+W.T
L=np.diag(W.sum(1))-W; M=L+0.05*np.eye(n); s=0.5; cs=math.sin(math.pi*s)/math.pi
def trp(X,p): return (np.linalg.eigvalsh(X)**p).sum()
S=[2]; v=5; beta=3.0
E=np.zeros((n,n)); E[2,2]=beta
Ms=M+E; Ev=np.zeros((n,n)); Ev[v,v]=beta
true=trp(Ms,-s)-trp(Ms+Ev,-s)
def integrand(t):
    R=np.linalg.inv(Ms+t*np.eye(n)); x=R[:,v]; r=x[v]; r2=x@x
    return t**(-s)*beta*r2/(1+beta*r)
num=cs*integrate.quad(integrand,0,np.inf,limit=400)[0]
print("soft grounding marginal  true=%.12f  formula=%.12f"%(true,num))
keep=[u for u in range(n) if u!=2]; keep2=[u for u in keep if u!=v]
Mh=M[np.ix_(keep,keep)]; true_h=trp(Mh,-s)-trp(M[np.ix_(keep2,keep2)],-s); iv=keep.index(v)
def ih(t):
    R=np.linalg.inv(Mh+t*np.eye(len(keep))); x=R[:,iv]; return t**(-s)*(x@x)/x[iv]
print("hard grounding marginal  true=%.12f  formula=%.12f"%(true_h,cs*integrate.quad(ih,0,np.inf,limit=400)[0]))

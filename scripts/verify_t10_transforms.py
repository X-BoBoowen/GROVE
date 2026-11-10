"""Direct amplification obstructions: signs, Schur closure, and noncommutation."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from fractions import Fraction as F


def test():
    pair=np.array([[3.,-1.],[-1.,3.]])
    tensor=np.kron(pair,pair)
    assert tensor[0,3]==1
    assert abs(np.linalg.eigvalsh(tensor)[0]-4)<1e-12
    kronsum=np.kron(pair,np.eye(2))+np.kron(np.eye(2),pair)
    assert abs(np.linalg.eigvalsh(kronsum)[0]-4)<1e-12
    assert np.max(kronsum-np.diag(np.diag(kronsum)))<=0
    assert 4/9 < 4/6 == 2/3
    a=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])
    assert (a@a)[0,2]==1
    rng=np.random.default_rng(1014)
    cases=0
    for n in range(3,11):
        for _ in range(10):
            edges=[(i,i+1,float(rng.uniform(.2,2))) for i in range(n-1)]
            edges += [(i,j,float(rng.uniform(.2,2))) for i in range(n) for j in range(i+2,n)
                      if rng.random()<.3]
            aa=laplacian(n,edges)+np.diag(rng.uniform(.01,1,n))
            split=n//2
            p,q,d=aa[:split,:split],aa[:split,split:],aa[split:,split:]
            invd=np.linalg.inv(d)
            assert np.min(invd)>=-1e-12
            sch=p-q@invd@q.T
            off=sch-np.diag(np.diag(sch))
            assert np.max(off)<=1e-12
            assert min(sch.sum(axis=1))>=-1e-12
            assert np.linalg.eigvalsh(sch)[0]>0
            cases+=1
    # Midpoint identity forced by singleton restriction, and its failure for
    # representative nonlinear maps; all are evaluated spectrally.
    for x,y in [(.1,1.),(1.,3.),(2.,8.)]:
        center=(x+y)/2
        for f in [lambda z:z*z, np.sqrt, lambda z:1-np.exp(-z)]:
            assert abs((f(x)+f(y))/2-f(center))>1e-5
        for c,d in [(1.,0.),(2.,3.),(0.,1.)]:
            assert abs((c*x+d+c*y+d)/2-(c*center+d))<1e-12
            assert (c*x+d)/(c*y+d)>=x/y-1e-12
    # Exact rational inverse-Hadamard-square noncommutation witness.
    inverse=np.array([[F(3,4),F(1,2),F(1,4)],
                      [F(1,2),F(1),F(1,2)],
                      [F(1,4),F(1,2),F(3,4)]],dtype=object)
    transformed=np.array([[F(2),F(-1,2),F(0)],
                          [F(-1,2),F(5,4),F(-1,2)],
                          [F(0),F(-1,2),F(2)]],dtype=object)
    assert np.array_equal((inverse*inverse)@transformed,np.eye(3,dtype=int))
    assert transformed[0,0]==2 and F(1)/(F(1,2)**2)==4
    assert np.linalg.eigvalsh(np.asarray(transformed,float))[0]>0
    print(f'PASS T10 transform obstructions: {cases} Schur checks; tensor/power signs exact; '
          'inverse-Hadamard witness exact rational; spectral/Schur tolerance=1e-12')


run('T10 transform obstructions', test)

"""Exact uniform-decoration spectral identity and concave ratio compression."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def test():
    rootgraph=laplacian(5,[(0,1,1.),(1,2,.7),(2,3,1.3),(3,4,.9),(4,0,1.1),(1,4,.3)])
    decorations=[laplacian(2,[(0,1,.1)]),
                 laplacian(4,[(0,1,1.),(1,2,.4),(2,3,2.)]),
                 laplacian(4,[(0,1,.7),(0,2,1.2),(0,3,.2),(1,2,.6)])]
    checks, ratio_checks, max_error=0,0,0.
    for hmat in decorations:
        h=len(hmat)-1
        full=np.zeros((5*(h+1),5*(h+1)))
        full[:5,:5]=rootgraph
        for u in range(5):
            indices=[u]+list(range(5+u*h,5+(u+1)*h))
            full[np.ix_(indices,indices)]+=hmat
        dmin=np.linalg.eigvalsh(hmat[1:,1:])[0]
        def g(mu):
            mat=hmat.copy(); mat[0,0]+=mu
            return np.linalg.eigvalsh(mat)[0]
        assert abs(g(0))<1e-12
        for s in subsets(5):
            if not 0<len(s)<5: continue
            mu=np.linalg.eigvalsh(grounded(rootgraph,s,np.inf))[0]
            value=np.linalg.eigvalsh(grounded(full,s,np.inf))[0]
            error=abs(value-g(mu))
            max_error=max(max_error,error)
            assert error<2e-11
            assert 0<g(mu)<dmin
            checks+=1
        for aa in [.001,.01,.1,1.]:
            for bb in [2.,5.,20.]:
                assert g(aa)/g(bb)>=aa/bb-1e-11
                for weight in [.1,.5,.9]:
                    assert g(weight*aa+(1-weight)*bb)>=weight*g(aa)+(1-weight)*g(bb)-1e-11
                ratio_checks+=1
    print(f'PASS T10 pendant decorations: {checks} all-set spectral identities, '
          f'{ratio_checks} ratio/concavity checks; max identity error={max_error:.3e}; tolerance=2e-11')


run('T10 pendant decorations', test)

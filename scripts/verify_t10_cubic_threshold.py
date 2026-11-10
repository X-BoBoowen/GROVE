"""Polynomial local-cut construction attaining 2/3 in the cubic half-budget regime."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def test():
    rng = np.random.default_rng(1013)
    graphs = [(4, [(i,j,1) for i in range(4) for j in range(i+1,4)]),
              (6, [(i,j,1) for i in range(3) for j in range(3,6)]),
              (8, [(i,i^(1<<j),1) for i in range(8) for j in range(3) if i<i^(1<<j)]),
              (10, [(i,(i+1)%5,1) for i in range(5)]+[(i,i+5,1) for i in range(5)]
                   +[(i+5,(i+2)%5+5,1) for i in range(5)])]
    checks, flips = 0, 0
    for n, edges in graphs:
        l=laplacian(n,edges)
        neighbors=[[v for v in range(n) if l[u,v]<0] for u in range(n)]
        for _ in range(10):
            color=rng.integers(0,2,n)
            rounds=0
            while True:
                bad=next((u for u in range(n) if sum(color[u]==color[v] for v in neighbors[u])>=2),None)
                if bad is None: break
                before=sum(color[u]!=color[v] for u,v,_ in edges)
                color[bad]^=1
                after=sum(color[u]!=color[v] for u,v,_ in edges)
                assert after>=before+1
                rounds+=1
            assert rounds<=len(edges)
            flips+=rounds
            for u in range(n):
                assert sum(color[u]==color[v] for v in neighbors[u])<=1
            big=max(([u for u in range(n) if color[u]==c] for c in [0,1]),key=len)
            for k in range(n//2,n):
                keep=set(big[:n-k]); selected=set(range(n))-keep
                value=np.linalg.eigvalsh(grounded(l,selected,np.inf))[0]
                optimum=max(np.linalg.eigvalsh(grounded(l,s,np.inf))[0]
                            for s in itertools.combinations(range(n),k))
                assert value>=2-1e-12 and value>=2*optimum/3-1e-12
                checks+=1
    # K4 at k=n/2 is a fixed NO cover instance for the promise reduction.
    assert not any(all(u in s or v in s for u,v,_ in graphs[0][1])
                   for s in itertools.combinations(range(4),2))
    print(f'PASS T10 cubic sharp threshold: {checks} exact-optimum comparisons, '
          f'{flips} improving color flips; tolerance=1e-12')


run('T10 cubic sharp threshold', test)

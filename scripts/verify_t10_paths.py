"""Parallel-subdivision gap, including anchors inside gadgets (T10)."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def constants(ell):
    ta = np.pi/(2*ell-1)
    tb = brentq(lambda t: 3*np.cos((ell-.5)*t)-np.cos(t/2), 0., ta,
                xtol=1e-15)
    return 4*np.sin(ta/2)**2, 4*np.sin(tb/2)**2, tb


def subdivide(n, edges, copies, ell):
    paths, new_edges, size = [], [], n
    for u, v, _ in edges:
        for _ in range(copies):
            path = [u]+list(range(size, size+ell-1))+[v]
            size += ell-1
            paths.append(path)
            new_edges.extend(zip(path[:-1], path[1:]))
    rr, cc, vv = [], [], []
    for u, v in new_edges:
        rr.extend([u, v, u, v]); cc.extend([u, v, v, u]); vv.extend([1., 1., -1., -1.])
    return coo_matrix((vv, (rr, cc)), shape=(size, size)).tocsr(), paths


def smallest(a, selected):
    keep = np.array([i for i in range(a.shape[0]) if i not in selected])
    aa = a[keep][:, keep]
    if len(keep) < 5:
        return np.linalg.eigvalsh(aa.toarray())[0]
    return eigsh(aa, k=1, which='SA', return_eigenvectors=False, tol=1e-11,
                 v0=np.ones(len(keep)))[0]


def test():
    rng = np.random.default_rng(1011)
    checks, trials, worst_eq = 0, 0, 0.
    for ell in [2, 3, 4, 8, 16, 64, 256]:
        a, b, theta = constants(ell)
        assert 0 < b < a <= 1+1e-14
        assert b/a > (2*np.arccos(1/3)/np.pi)**2
        h = ell-1
        nd = np.diag(np.r_[1., np.full(h-1, 2.)])
        if h > 1:
            nd += np.diag(np.full(h-1, -1.), 1)+np.diag(np.full(h-1, -1.), -1)
        assert abs(np.linalg.eigvalsh(nd)[0]-a) < 2e-12
        residual = 3*np.cos((ell-.5)*theta)-np.cos(theta/2)
        assert abs(residual) < 2e-12
        for endpoints in [(1., 1.), (1., 0.), (0., 1.)]:
            x, y = endpoints
            internal = np.array([(x*np.sin((ell-j)*theta)+y*np.sin(j*theta))/np.sin(ell*theta)
                                 for j in range(1, ell)])
            assert min(internal) >= 0 and max(internal) <= 2
            full = np.r_[x, internal, y]
            assert np.max(abs(2*internal-full[:-2]-full[2:]-b*internal)) < 2e-12
            if x == y == 1:
                assert min(internal) >= 1-1e-12
        checks += 1
    limit = (2*np.arccos(1/3)/np.pi)**2
    assert abs(constants(256)[1]/constants(256)[0]-limit) < 2e-6
    graphs = [(4, [(i, j, 1) for i in range(4) for j in range(i+1, 4)]),
              (6, [(i, j, 1) for i in range(3) for j in range(3, 6)])]
    for n, edges in graphs:
        for ell in [2, 3, 5]:
            a, b, theta = constants(ell)
            budget, copies = 3, 24  # M*(ell-1)>=8k; all nodes eligible.
            l, paths = subdivide(n, edges, copies, ell)
            cover = {0, 1, 2}
            assert all(u in cover or v in cover for u, v, _ in edges)
            value = smallest(l, cover)
            lower = a/(1+2*(1+a)/(3*copies))
            assert value >= lower-2e-10
            # Deliberately cover root-only, internal-only, and mixed choices.
            candidates = [set([0, n, n+1]), set([n, n+1, n+ell]), set([0, 1, n])]
            candidates += [set(rng.choice(l.shape[0], budget, replace=False)) for _ in range(8)]
            for selected in candidates:
                selected = set(map(int, selected))
                assert len(selected) == budget
                rootset = selected & set(range(n))
                uncovered = [(u, v) for u, v, _ in edges if u not in rootset and v not in rootset]
                if not uncovered:
                    continue
                u, v = uncovered[0]
                z = np.zeros(l.shape[0]); z[[u, v]] = 1.
                for path in paths:
                    x, y = z[path[0]], z[path[-1]]
                    for j in range(1, ell):
                        z[path[j]] = (x*np.sin((ell-j)*theta)+y*np.sin(j*theta))/np.sin(ell*theta)
                mass = np.dot(z[n:], z[n:])
                energy = z@(l@z)
                discrepancy = abs(energy-b*mass)/max(1., energy)
                worst_eq = max(worst_eq, discrepancy)
                assert discrepancy < 2e-12
                assert mass >= copies*(ell-1)-2e-10
                assert max(z) <= 2
                modified = z.copy(); modified[list(selected)] = 0
                changed_energy = modified@(l@modified)
                changed_norm = modified@modified
                assert changed_energy <= energy+8*budget+2e-10
                assert changed_norm >= z@z-4*budget-2e-10
                quotient = changed_energy/changed_norm
                upper = b+24*budget/(copies*(ell-1))
                actual = smallest(l, selected)
                assert actual <= quotient+2e-9
                assert quotient <= upper+2e-10
                trials += 1
    print(f'PASS T10 parallel paths: {checks} analytic grids, {trials} all-node anchor sets; '
          f'max trial energy identity error={worst_eq:.3e}; tolerance=2e-9; '
          f'limiting ratio={limit:.10f}')


run('T10 parallel paths', test)

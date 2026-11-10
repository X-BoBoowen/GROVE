"""T12 inexact-column energy budgets and higher-power residual certificates."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def inverse(a, selected):
    u = [i for i in range(len(a)) if i not in selected]
    r = np.zeros_like(a); r[np.ix_(u, u)] = np.linalg.inv(a[np.ix_(u, u)])
    return r


def energy(e, a):
    return np.sqrt(max(0, np.sum((e@a)*e)))


def test():
    rng = np.random.default_rng(1214)
    updates = chains = 0
    worst_ratio = 0.0
    for a in fixtures()[:-1]:
        n, d = len(a), 1024
        lo, hi = np.linalg.eigvalsh(a)[[0, -1]]
        c0 = np.linalg.cholesky(a).T
        for amplitude in [0., 1e-7, 1e-4, 1e-2]:
            g, h = [rng.normal(size=(d, n))/np.sqrt(d) for _ in range(2)]
            lefts = [g@c0, h]
            r = np.linalg.inv(a); selected = set()
            yy = [f@r+rng.normal(size=(d, n))*1e-5 for f in lefts]
            bounds = [energy(y-f@r, a) for y, f in zip(yy, lefts)]
            for step in range(3):
                active = [u for u in range(n) if u not in selected]
                v = max(active, key=lambda u: np.sum(yy[1][:, u]**2)/np.sum(yy[0][:, u]**2))
                x = r[:, v]
                perturb = np.zeros(n); perturb[active] = rng.normal(size=len(active))*amplitude
                xt = x+perturb
                assert xt[v] > 0
                au = a[np.ix_(active, active)]
                rhs = np.zeros(len(active)); rhs[active.index(v)] = 1
                rho = np.linalg.norm(au@xt[active]-rhs)
                normalized = xt/xt[v]-x/x[v]
                exact_norm = energy(perturb[None, :], a)**2-perturb[v]**2/x[v]
                lhs = energy(normalized[None, :], a)**2
                assert abs(lhs-exact_norm/xt[v]**2) < 3e-10*max(1, lhs)
                assert np.sqrt(lhs) <= rho/(np.sqrt(lo)*xt[v])+2e-10
                for i in range(2):
                    col = yy[i][:, v].copy()
                    noise = rng.normal(size=yy[i].shape)*1e-9
                    noise[:, list(selected|{v})] = 0
                    bounds[i] += np.linalg.norm(col)*rho/(np.sqrt(lo)*xt[v])+energy(noise, a)
                    yy[i] -= np.outer(col, xt)/xt[v]
                    yy[i] += noise
                selected.add(v); r = inverse(a, selected)
                active = [u for u in range(n) if u not in selected]
                for i, (y, f) in enumerate(zip(yy, lefts)):
                    actual = energy(y-f@r, a)
                    assert actual <= bounds[i]+2e-9
                    worst_ratio = max(worst_ratio, actual/bounds[i])
                    ideal = f@r
                    target = np.diag(r if i == 0 else r@r)
                    zeta = np.max(np.abs(np.sum(ideal[:, active]**2, axis=0)/target[active]-1))
                    ee = bounds[i]*(1 if i == 0 else np.sqrt(hi))
                    rel = np.max(np.abs(np.sum(y[:, active]**2, axis=0)/target[active]-1))
                    assert rel <= zeta+2*np.sqrt(1+zeta)*ee+ee*ee+2e-9
                updates += 1
        # Residual chains are checked at nonempty sets, including fresh independent perturbations.
        for selected in [set(), {0}, {0, 2}]:
            active = [u for u in range(n) if u not in selected]
            au = a[np.ix_(active, active)]
            r = np.linalg.inv(au)
            for odd in [False, True]:
                f = rng.normal(size=(128, n))/np.sqrt(128)
                if odd: f = f@c0
                f = f[:, active]
                prev, bound = f.copy(), 0.
                for ell in range(1, 6):
                    ideal = f@np.linalg.matrix_power(r, ell)
                    current = ideal+rng.normal(size=ideal.shape)*1e-6
                    rho = np.linalg.norm(current@au-prev, 'fro')
                    bound = (bound+rho)/lo
                    assert np.linalg.norm(current-ideal, 'fro') <= bound+1e-10
                    j = 2*ell-int(odd)
                    target = np.diag(np.linalg.matrix_power(r, j))
                    zeta = max(.0, np.max(np.abs(np.sum(ideal**2, axis=0)/target-1)))
                    ee = hi**(ell-.5*odd)*bound
                    rel = np.max(np.abs(np.sum(current**2, axis=0)/target-1))
                    assert rel <= zeta+2*np.sqrt(1+zeta)*ee+ee*ee+2e-9
                    prev = current; chains += 1
    for bar in [.001, .01, .1, .5, .99]:
        zeta, ee = bar/2, bar/8
        assert zeta+2*np.sqrt(1+zeta)*ee+ee*ee <= bar
    print(f'PASS T12 residuals: {updates} adaptive inexact updates and {chains} chain levels; '
          f'max actual/energy-envelope={worst_ratio:.6f}; normalized-column identity tolerance 3e-10; '
          'moment error and explicit budget checks pass')


run('T12 maintained-sketch residual budgets', test)

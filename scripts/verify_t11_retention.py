"""Moment-retention ratios, weighted return integrals, and electrical identities."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from scipy.integrate import quad
from scipy.special import gamma
from math import comb


def test():
    rng = np.random.default_rng(1104)
    cases = integrals = 0
    worst = 0.0
    for base in fixtures()[:-1]:
        n = len(base)
        vals, vecs = np.linalg.eigh(base)
        r0 = np.linalg.inv(base)
        killing = base.sum(axis=1)
        lap = np.block([[np.array([[killing.sum()]]), -killing[None, :]],
                        [-killing[:, None], base]])
        pinv = np.linalg.pinv(lap)
        for size in [1, 2]:
            selected = np.sort(rng.choice(n, size, replace=False))
            ulist = np.array([i for i in range(n) if i not in selected])
            current = base[np.ix_(ulist, ulist)]
            eig, vectors = np.linalg.eigh(current)
            rt = np.zeros_like(base)
            rt[np.ix_(ulist, ulist)] = np.linalg.inv(current)
            drop = r0[:, selected]@np.linalg.solve(r0[np.ix_(selected, selected)], r0[selected, :])
            assert np.max(np.abs(r0-drop-rt)) < 2e-9
            hitting = -np.linalg.solve(current, base[np.ix_(ulist, selected)])
            for ix, u in enumerate(ulist):
                alpha = drop[u, u]/r0[u, u]
                hit_total = hitting[ix].sum()
                alpha_from_return = hitting[ix]@r0[selected, u]/r0[u, u]
                assert abs(alpha-alpha_from_return) < 2e-9
                assert alpha <= hit_total+2e-9 and hit_total <= 1+2e-9
                if size == 1:
                    v = selected[0]
                    product = r0[u, v]**2/(r0[u, u]*r0[v, v])
                    assert abs(alpha-product) < 2e-9
                    def resistance(i, j):
                        return pinv[i, i]+pinv[j, j]-2*pinv[i, j]
                    du, dv = resistance(u+1, 0), resistance(v+1, 0)
                    duv = resistance(u+1, v+1)
                    electrical = (du+dv-duv)**2/(4*du*dv)
                    assert abs(alpha-electrical) < 2e-9
                w0, wt = vecs[u]**2, vectors[ix]**2
                for j in [1, 2, 3, 4, 5]:
                    m0 = np.sum(w0*vals**(-j))
                    mt = np.sum(wt*eig**(-j))
                    retention = mt/m0
                    assert 0 < retention <= 1+2e-9
                    high = np.sum(w0*vals**(-(2*j-1)))
                    cs = np.sqrt(alpha*gamma(2*j-1)*high*r0[u, u])/(gamma(j)*m0)
                    condition_bound = np.sqrt(comb(2*j-2, j-1)*alpha*(vals[-1]/vals[0])**(j-1))
                    assert 1-retention <= min(1, cs)+2e-9
                    assert cs <= condition_bound+2e-9*max(1, condition_bound)
                    # Loss is exactly the mass of paths killed by the new set.
                    def integrand(t):
                        original = np.dot(w0, np.exp(-t*vals))
                        survivor = np.dot(wt, np.exp(-t*eig))
                        return t**(j-1)*(original-survivor)/gamma(j)
                    integral, err = quad(integrand, 0, np.inf, epsabs=1e-8, epsrel=1e-10)
                    discrepancy = abs(integral-(m0-mt))/max(1, m0)
                    worst = max(worst, discrepancy)
                    assert discrepancy < 3e-8
                    initial_error = .01*m0
                    assert abs(initial_error/mt-.01/retention) < 1e-10/retention
                    lo_now, up_initial = .99*mt, 1.01*m0
                    assert lo_now/up_initial <= retention+1e-12
                    absolute_error = .002*mt
                    assert absolute_error <= .003*lo_now
                    assert absolute_error/mt <= .003
                    cases += 1
                    integrals += 1
    ratios = []
    for t in [.1, .01, .001, .0001, .00001]:
        a = np.array([[2*t, -t], [-t, 1+t]])
        r = np.linalg.inv(a)
        rho1 = 1/(1+t)/r[1, 1]
        rho3 = (1+t)**(-3)/np.linalg.matrix_power(r, 3)[1, 1]
        assert abs(rho1-(2+t)/(2*(1+t))) < 1e-10
        assert abs(r[1, 0]/r[0, 0]-t/(1+t)) < 1e-10
        ratios.append(rho3)
    assert ratios[-1] < 1e-3 and np.all(np.diff(ratios) < 0)
    print(f'PASS T11 retention: {cases} moment checks and {integrals} return-loss integrals; '
          f'max scaled integral error={worst:.3e} < 3e-8; higher-moment control rho3={ratios[-1]:.3e}')


run('T11 moment retention', test)

"""Fresh-round solve counts, precision recurrence, and p=1 specialization."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_phase3_utils.py').read_text(), '_phase3_utils.py', 'exec'))


def test():
    rng = np.random.default_rng(7103)
    cases = 0
    for p in [1, 2, 3, 4, 8, 16, 32, 64]:
        j, depth = p+1, (p+2)//2
        assert depth <= p
        for epsilon in [.01, .1, .5]:
            n, budget, delta = 100, 20, .05
            zeta = epsilon/(32*p)
            logarithm = np.log(12*n*j*budget/delta)
            d = int(np.ceil(128/zeta**2*logarithm))
            solves = 2*budget*d*depth
            assert solves <= 2*budget*(128*1024*p*p/epsilon**2*logarithm+1)*p
            for kappa in [1., 2., 10., 1000.]:
                m = np.sqrt(6*n*budget/delta)
                log_inverse_xi = np.log(64*depth*(m+1)/zeta)+2*depth*np.log(kappa)
                assert log_inverse_xi > 0
                # Fixed-p SDDM precision factors, including the extra p factor.
                upper_log = p*(np.log(64*depth*(m+1)/zeta)+2*np.log(kappa))
                assert log_inverse_xi <= upper_log+1e-12
                assert p*p*depth*depth <= p**4
                # Stable scaled worst-case residual recurrence, without
                # underflow at the very small prescribed tolerances.
                scaled_error = sum(kappa**(i-depth) for i in range(1, depth+1))/depth
                assert scaled_error <= 1+1e-12
            if p == 1:
                assert j == 2 and depth == 1 and solves == 2*budget*d
            cases += 1
    # Actual perturbed energy-accurate solve chains for modest condition numbers.
    worst_envelope = 0.
    for p in [1, 2, 3, 5]:
        depth = (p+2)//2
        a = np.array([[.8, -.1, 0], [-.1, .75, -.1], [0, -.1, .8]])
        a /= np.linalg.eigvalsh(a)[-1]
        kappa = 1/np.linalg.eigvalsh(a)[0]
        zeta, rows = .3/(32*p), 8
        c = np.linalg.cholesky(a).T
        for start in [rng.normal(size=(rows, 3))/np.sqrt(rows),
                      rng.normal(size=(rows, 3))@c/np.sqrt(rows)]:
            m = np.linalg.norm(start, 'fro')
            xi = zeta/(64*depth*kappa**(2*depth)*(m+1))
            approximate, exact = start.copy(), start.copy()
            envelope = 0.
            for ell in range(1, depth+1):
                rhs = approximate.copy()
                target = np.linalg.solve(a, rhs.T).T
                noise = rng.normal(size=target.shape)
                for row in range(rows):
                    target_energy = np.sqrt(target[row]@a@target[row])
                    noise_energy = np.sqrt(noise[row]@a@noise[row])
                    noise[row] *= .8*xi*target_energy/noise_energy
                approximate = target+noise
                exact = np.linalg.solve(a, exact.T).T
                residual = np.linalg.norm(approximate@a-rhs, 'fro')
                error = np.linalg.norm(approximate-exact, 'fro')
                envelope = kappa*(envelope+residual)
                assert residual <= zeta/(64*depth*kappa**depth)+1e-13
                assert error <= envelope+1e-13
                assert error <= zeta/64+1e-13
                worst_envelope = max(worst_envelope, error/envelope)
    print(f'PASS T7b complexity: {cases} parameter regimes; f(p)=p^4 with precision counted; '
          f'energy-solve chains max actual/envelope={worst_envelope:.5f}; tolerance=1e-13; p=1 two-family specialization passes')


run('T7b complexity', test)

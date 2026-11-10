"""Masked seed inverse: one-sided outliers, polynomial bound, and PCG."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def root(a, power):
    w, v = np.linalg.eigh(a)
    return (v*w**power)@v.T


def pcg_errors(a, bpre, rhs):
    true = np.linalg.solve(a, rhs)
    den = np.sqrt(true@a@true)
    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = bpre@r
    p = z.copy()
    rz = r@z
    errors = [1.0]
    for _ in range(len(a)):
        ap = a@p
        step = rz/(p@ap)
        x += step*p
        r -= step*ap
        e = x-true
        errors.append(np.sqrt(max(0, e@a@e))/den)
        if np.linalg.norm(r) < 1e-14*np.linalg.norm(rhs):
            break
        z = bpre@r
        new = r@z
        p = z+(new/rz)*p
        rz = new
    return errors


def test():
    rng = np.random.default_rng(1101)
    count = 0
    max_violation = 0.0
    for n in [8, 13, 20]:
        edges = [(i, i+1, rng.uniform(.2, 2)) for i in range(n-1)]
        edges += [(i, j, rng.uniform(.2, 2)) for i in range(n)
                  for j in range(i+2, n) if rng.random() < .2]
        lap = laplacian(n, edges)
        # Q maps charge-neutral sources to voltages relative to the ground.
        q = np.column_stack([-np.ones(n-1), np.eye(n-1)])
        grounded_inverse = np.linalg.inv(lap[1:, 1:])
        assert np.allclose(q@np.linalg.pinv(lap)@q.T, grounded_inverse,
                           rtol=3e-9, atol=3e-9)
        for sigma in [0, .3]:
            a0 = lap[1:, 1:]+sigma*np.eye(n-1)
            inv = np.linalg.inv(a0)
            inverse_root = root(a0, -.5)
            orth, _ = np.linalg.qr(rng.normal(size=a0.shape))
            lo, hi = .75, 1.4
            b0 = inverse_root@((orth*np.linspace(lo, hi, n-1))@orth.T)@inverse_root
            for t in range(5):
                selected = rng.choice(n-1, t, replace=False)
                u = np.array([i for i in range(n-1) if i not in selected])
                a = a0[np.ix_(u, u)]
                b = b0[np.ix_(u, u)]
                ar = root(a, .5)
                h = ar@b@ar
                eig = np.linalg.eigvalsh(h)
                assert eig[0] >= lo-3e-9
                out = eig[eig > hi+1e-9]
                assert len(out) <= t
                exact_b = inv[np.ix_(u, u)]
                kk = ar@exact_b@ar-np.eye(len(u))
                assert np.linalg.eigvalsh(kk)[0] >= -3e-9
                assert np.linalg.matrix_rank(kk, tol=3e-8) <= t
                padded = np.zeros_like(a0)
                padded[np.ix_(u, u)] = a
                padded[selected, selected] = 1
                assert np.linalg.matrix_rank(padded-a0, tol=3e-9) <= 2*t
                rho = (np.sqrt(hi/lo)-1)/(np.sqrt(hi/lo)+1)
                for degree in [0, 1, 2, 4, 8]:
                    coeff = np.r_[np.zeros(degree), 1.]
                    x = (hi+lo-2*eig)/(hi-lo)
                    x0 = (hi+lo)/(hi-lo)
                    poly = np.polynomial.chebyshev.chebval(x, coeff)
                    poly /= np.polynomial.chebyshev.chebval(x0, coeff)
                    for mu in out:
                        poly *= 1-eig/mu
                    # Outlier roots are exact up to the eigensolver's roundoff.
                    assert np.max(np.abs(poly)) <= 2*rho**degree+3e-8
                for _ in range(2):
                    rhs = rng.normal(size=len(u))
                    errors = pcg_errors(a, b, rhs)
                    for step, err in enumerate(errors):
                        if step >= len(out):
                            bound = 2*rho**(step-len(out))
                            max_violation = max(max_violation, err-bound)
                            assert err <= bound+3e-9
                    exact_errors = pcg_errors(a, exact_b, rhs)
                    assert exact_errors[min(t+1, len(exact_errors)-1)] < 3e-9
                # Residual and energy conversion for an independently perturbed solve.
                rhs = rng.normal(size=len(u))
                truth = np.linalg.solve(a, rhs)
                error = rng.normal(size=len(u))*1e-5
                residual_ratio = np.linalg.norm(a@error)/np.linalg.norm(rhs)
                energy_ratio = np.sqrt(error@a@error/(truth@a@truth))
                condition = np.linalg.cond(a)
                assert energy_ratio <= np.sqrt(condition)*residual_ratio+1e-12
                assert residual_ratio <= np.sqrt(condition)*energy_ratio+1e-12
                count += 1
    print(f'PASS T11 PCG: {count} seed/grounding systems; one-sided outliers, padded rank, '
          f'Q identity, exact-preconditioner termination; max CG bound violation={max_violation:.3e} < 3e-9')


run('T11 masked PCG', test)

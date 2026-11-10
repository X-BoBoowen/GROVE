"""Tight positive-word upper bounds, including noisy moment intervals."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_phase3_utils.py').read_text(), '_phase3_utils.py', 'exec'))


def test():
    rng = np.random.default_rng(811)
    cases = 0
    worst = 0.0
    matrices = fixtures()
    for n in [3, 6, 9]:
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        matrices.append(q @ np.diag(np.geomspace(.3, 4, n)) @ q.T)
    for x in matrices:
        for v in range(len(x)):
            rs = np.array(moments(x, v, 13))
            for beta in [.2, 3, np.inf]:
                c = 1 / ((0 if np.isinf(beta) else 1/beta) + rs[0])
                a = c * rs[1:]
                for p in [2, 3, 4, 8, 12]:
                    exact, _, _ = positive_integer_marginal(x, v, beta, p)
                    d2 = max(0.0, a[1] - a[0]**2)
                    u1 = p*a[p-1] - (p-1)*a[0]**p
                    u2 = u1 - p*(p-2)*d2*a[0]**(p-2)
                    scale = max(1.0, abs(exact), p*a[p-1])
                    violations = [exact-u2, u2-u1, u1-p*a[p-1], a[p-1]-u2]
                    worst = max(worst, max(violations) / scale)
                    assert max(violations) <= 3e-8*scale
                    if p <= 3:
                        assert abs(exact-u2) <= 3e-8*scale
                    if p == 2:
                        assert abs(exact-u1) <= 3e-8*scale
                    # Arbitrarily correlated valid intervals need no realizability.
                    lower = a * (1-rng.uniform(0, .2, len(a)))
                    upper = a * (1+rng.uniform(0, .2, len(a)))
                    vminus = max(0.0, lower[1]-upper[0]**2)
                    certified = p*upper[p-1] - (p-1)*lower[0]**p
                    certified -= p*(p-2)*vminus*lower[0]**(p-2)
                    assert exact <= certified + 3e-8*scale
                    cases += 1
    # Rank-one hard endpoint: the old bound is exactly p times too large.
    for p in [2, 3, 4, 8, 16]:
        z = 1.7
        old = p*z**p
        new = old-(p-1)*z**p
        assert abs(new/z**p-1) < 1e-12
    print(f'PASS tighter upper bounds: {cases} exact/interval cases; max scaled violation={worst:.3e} '
          '< 3e-8; p=2/3 exactness and rank-one improvement pass')


run('Tighter marginal upper bounds', test)

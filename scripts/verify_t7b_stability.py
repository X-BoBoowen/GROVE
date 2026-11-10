"""T7b polynomial perturbation bound, including inconsistent moment estimates."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_phase3_utils.py').read_text(), '_phase3_utils.py', 'exec'))
from decimal import Decimal, localcontext


def decimal_marginal(rs, offset, p):
    aa = [Decimal(1)] + [r/(offset+rs[0]) for r in rs[1:p+1]]
    qq = [Decimal(0)]*(p+1)
    for j in range(1, p+1):
        qq[j] = j*aa[j]-sum((qq[i]*aa[j-i] for i in range(1, j)), Decimal(0))
    return qq[p], aa


def test():
    rng = np.random.default_rng(7101)
    counts, worst = 0, 0.
    # Matrix checks of the positive reciprocal coefficient identity.
    for a in fixtures():
        v = len(a)//2
        for penalty in [.1, 3., np.inf]:
            for p in range(1, 9):
                rs = moments(a, v, p+1)
                qq, dd, _ = positive_integer_marginal(a, v, penalty, p)
                aa = np.r_[1., rs[1:]/((0 if penalty == np.inf else 1/penalty)+rs[0])]
                for j in range(1, p+1):
                    rhs = dd[j]+sum(aa[i]*dd[j-i] for i in range(1, j))
                    assert abs(aa[j]-rhs) <= 3e-9*max(1., aa[j])
                assert qq >= aa[p]-3e-9*max(1., qq)
    # High precision prevents the test recurrence's own floating-point
    # cancellation from being confused with input-moment perturbations.
    with localcontext() as ctx:
        ctx.prec = 100
        eta = Decimal('.4')
        for zs in [['1'], ['.001', '.1', '.7', '1'], ['.91', '.94', '.98', '1']]:
            z = list(map(Decimal, zs))
            w = [Decimal(i+1) for i in range(len(z))]
            w = [x/sum(w) for x in w]
            for p in [1, 2, 3, 4, 5, 8, 12, 16, 32, 64]:
                rs = [sum((wi*zi**j for wi, zi in zip(w, z)), Decimal(0)) for j in range(1, p+2)]
                zeta = eta/(16*p)
                bound = (1+4*zeta/(1-zeta))**p-1
                assert bound <= eta
                signs = (list(itertools.product([-1, 1], repeat=p+1)) if p <= 4
                         else rng.uniform(-1, 1, (30, p+1)))
                for offset in [Decimal(0), Decimal('.2'), Decimal('10')]:
                    exact, aa = decimal_marginal(rs, offset, p)
                    assert exact >= aa[p]-Decimal('1e-75')
                    assert exact <= p*aa[p]+Decimal('1e-75')
                    for sign in signs:
                        perturbed = [r*(1+zeta*Decimal(str(s))) for r, s in zip(rs, sign)]
                        approx, _ = decimal_marginal(perturbed, offset, p)
                        error = abs(approx/exact-1)
                        assert error <= bound+Decimal('1e-70'), (p, error, bound)
                        assert approx > 0
                        counts += 1
                        worst = max(worst, float(error/bound))
        # Explicit arbitrary normalized-coefficient errors and the coefficient
        # bound |[z^j](E/F)| <= 2 alpha a_j.
        p = 20
        rs = [sum((wi*zi**j for wi, zi in zip(w, z)), Decimal(0)) for j in range(1, p+2)]
        _, aa = decimal_marginal(rs, Decimal('.2'), p)
        dd = [Decimal(0)]*(p+1)
        for j in range(1, p+1):
            dd[j] = aa[j]-sum((aa[i]*dd[j-i] for i in range(1, j)), Decimal(0))
            assert dd[j] >= -Decimal('1e-75')
        alpha = Decimal('.002')
        ee = [Decimal(0)]+[alpha*aa[j]*Decimal(str(rng.uniform(-1, 1))) for j in range(1, p+1)]
        for j in range(1, p+1):
            hh = ee[j]-sum((ee[i]*dd[j-i] for i in range(1, j)), Decimal(0))
            assert abs(hh) <= 2*alpha*aa[j]+Decimal('1e-75')
    print(f'PASS T7b polynomial stability: {counts} independent perturbations through p=64; '
          f'max actual/proved={worst:.5f}; Decimal tolerance=1e-70; matrix reciprocal tolerance=3e-9')


run('T7b polynomial stability', test)

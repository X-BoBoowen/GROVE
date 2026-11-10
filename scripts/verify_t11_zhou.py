"""Low-order finite-removal polynomials and first-order Zhou comparison."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_phase3_utils.py').read_text(), '_phase3_utils.py', 'exec'))
from fractions import Fraction


def polynomial(a, z, p):
    a1, a2, a3, a4 = a
    if p == 2:
        return 2*z*a2-z*z*a1*a1
    if p == 3:
        return 3*z*a3-3*z*z*a1*a2+z**3*a1**3
    return 4*z*a4-z*z*(4*a1*a3+2*a2*a2)+4*z**3*a1*a1*a2-z**4*a1**4


def test():
    cases = 0
    worst = 0.0
    for xmat in fixtures():
        eig, vec = np.linalg.eigh(xmat)
        r = np.linalg.inv(xmat)
        adjacency = np.diag(np.diag(xmat))-xmat
        phi = vec[:, 0]
        scores = 2*phi*(adjacency@phi)
        assert np.allclose(scores, 2*(np.diag(xmat)-eig[0])*phi**2, atol=2e-10)
        for v in range(len(r)):
            rs = moments(xmat, v, 5)
            a = rs[1:]/rs[0]
            d = np.outer(r[:, v], r[:, v])/r[v, v]
            for p in [2, 3, 4]:
                first = p*a[p-1]
                weights = vec[v]**2
                spectral = p/rs[0]*sum(weights*eig**(-p-1))
                assert abs(first-spectral) <= 2e-9*max(1, abs(first))
                if xmat[v, v]-eig[0] > 1e-9:
                    term = p*eig[0]**(-p-1)*scores[v]/(2*rs[0]*(xmat[v, v]-eig[0]))
                    tail = p/rs[0]*sum(weights[1:]*eig[1:]**(-p-1))
                    assert abs(first-term-tail) < 2e-9*max(1, abs(first))
                    if len(eig) > 1:
                        assert tail <= p/rs[0]*(1-weights[0])*eig[1]**(-p-1)+2e-9
                for z in [.001, .02, .2, .6, 1.]:
                    exact = np.trace(np.linalg.matrix_power(r, p)-np.linalg.matrix_power(r-z*d, p))
                    expected = polynomial(a, z, p)
                    scale = max(1, abs(exact), abs(first))
                    err = abs(exact-expected)/scale
                    worst = max(worst, err)
                    assert err < 2e-9
                    remainder = z*first-exact
                    bound = p*(p-1)/2*z*z*a[0]**2*np.linalg.norm(r, 2)**(p-2)
                    assert -2e-9*scale <= remainder <= bound+2e-9*scale
                    if z < 1:
                        beta = z/((1-z)*rs[0])
                        inv_soft = np.linalg.inv(xmat+beta*np.outer(np.eye(len(r))[v], np.eye(len(r))[v]))
                        assert np.max(np.abs(inv_soft-(r-z*d))) < 2e-9*np.linalg.norm(r, 2)
                    cases += 1
                beta = 1e-6/np.linalg.norm(r, 2)
                soft = np.linalg.inv(xmat+beta*np.outer(np.eye(len(r))[v], np.eye(len(r))[v]))
                difference = np.trace(np.linalg.matrix_power(r, p)-np.linalg.matrix_power(soft, p))/beta
                soft_first = p*rs[p]
                assert abs(difference-soft_first) < 2e-5*max(1, soft_first)
    # Rational counterexample: p=2,3,4 choose the first vertex; Zhou chooses the middle.
    xmat = np.array([[1., -.9, 0], [-.9, 3, -.5], [0, -.5, 1]])
    eig, vec = np.linalg.eigh(xmat)
    phi = np.abs(vec[:, 0])
    score = 2*phi*((np.diag(np.diag(xmat))-xmat)@phi)
    assert np.argmax(score) == 1
    selected_spectra = [np.linalg.eigvalsh(updated(xmat, v, np.inf)) for v in range(3)]
    assert selected_spectra[0][0] < 1 and selected_spectra[2][0] < .7
    detinv, traceinv = Fraction(4, 11), Fraction(16, 11)
    powers = [Fraction(2), traceinv]
    for p in range(2, 5):
        powers.append(traceinv*powers[-1]-detinv*powers[-2])
    assert powers[2:] == [Fraction(168, 121), Fraction(1984, 1331), Fraction(24352, 14641)]
    for p in [2, 3, 4]:
        objectives = [sum(w**(-p)) for w in selected_spectra]
        assert np.argmin(objectives) == 0
        assert powers[p] < 2
        assert abs(objectives[0]-float(powers[p])) < 2e-10
    print(f'PASS T11 Zhou comparison: {cases} finite-removal polynomial checks; max scaled error={worst:.3e} '
          '< 2e-9; spectral weights, soft derivatives, remainder bounds and exact-rational tradeoff control pass')


run('T11 Zhou expansion', test)

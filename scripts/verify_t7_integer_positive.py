"""Noncancelling integer marginal representation and dimension-free magnitude bounds."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))


def test():
    rng = np.random.default_rng(701)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    generic = q@np.diag([.4, .7, 1.3, 2, 6])@q.T
    cases, worst = 0, 0.0
    for a in fixtures()+[generic]:
        for v in range(len(a)):
            for shift in [0, .4]:
                at = a+shift*np.eye(len(a))
                rs = moments(at, v, 9)
                for penalty in [.05, 3, np.inf]:
                    c = 1/((0 if penalty == np.inf else 1/penalty)+rs[0])
                    for order in range(1, 9):
                        positive, d, h = positive_integer_marginal(at, v, penalty, order)
                        scalar = moment_marginal(rs, penalty, order)
                        target = trace_function(at, lambda x: x**(-order))
                        target -= trace_function(updated(at, v, penalty), lambda x: x**(-order))
                        scale = max(1, abs(target))
                        error = max(abs(positive-target), abs(scalar-target))/scale
                        worst = max(worst, error)
                        assert error < 3e-8
                        assert np.all(d >= 0) and np.all(h >= 0)
                        lower = c*rs[order]
                        assert lower-3e-8*scale <= target <= order*lower+3e-8*scale
                        if order == 2:
                            # Add a zero atom for finite penalty and evaluate mu2+variance.
                            vals, vec = np.linalg.eigh(at)
                            z = np.r_[0, 1/vals]
                            atom = 0 if penalty == np.inf else (1/penalty)*c
                            prob = np.r_[atom, c*vec[v]**2/vals]
                            mean = np.sum(prob*z)
                            variance = np.sum(prob*(z-mean)**2)
                            noncancelling = np.sum(prob*z*z)+variance
                            assert abs(noncancelling-target) < 3e-8*scale
                            assert (c*rs[1])**2 <= c*rs[2]+1e-10*scale
                        cases += 1
    print(f"PASS T7 integer positivity: {cases} power/shift/node checks; max scaled error={worst:.3e} < 3e-8; "
          "positive recurrence, a_m<=q_m<=m*a_m, and q2 variance form pass")


run("T7 integer positivity", test)

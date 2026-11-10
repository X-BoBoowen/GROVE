"""Scalar resolvent lemma: soft/hard drops, 0<s<1 integrals and s=1."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
from scipy.integrate import quad


def test():
    worst, cases = 0.0, 0
    for a in fixtures():
        v = len(a)//2
        for beta in [.2, 3, np.inf]:
            b = updated(a, v, beta)
            for t in [0, .01, 1, 10]:
                x = np.linalg.solve(a+t*np.eye(len(a)), np.eye(len(a))[:, v])
                r1, r2 = x[v], x@x
                h = 1e-5*max(1, t)
                derivative = (np.linalg.inv(a+(t+h)*np.eye(len(a)))[v, v]
                              - np.linalg.inv(a+(t-h)*np.eye(len(a)))[v, v])/(2*h)
                assert abs(derivative+r2) <= 2e-7*max(1, r2)
                formula = r2/r1 if beta == np.inf else beta*r2/(1+beta*r1)
                target = trace_function(a+t*np.eye(len(a)), lambda x: 1/x)
                target -= trace_function(b+t*np.eye(len(b)), lambda x: 1/x)
                assert abs(target-formula) <= 2e-10*max(1, target)
            for s in [.1, .5, .9]:
                # t=u/(1-u), factoring both endpoint singularities explicitly.
                def integrand(u):
                    h = (1-u)*a+u*np.eye(len(a))
                    penalty = np.inf if beta == np.inf else beta*(1-u)
                    return scalar_resolvent_drop(h, v, penalty)
                value, _ = quad(integrand, 0, 1, weight="alg", wvar=(-s, s-1),
                                epsabs=2e-10, epsrel=2e-10, limit=200)
                value *= np.sin(np.pi*s)/np.pi
                target = trace_function(a, lambda x: x**(-s))-trace_function(b, lambda x: x**(-s))
                err = abs(value-target)/max(1, abs(target))
                worst = max(worst, err)
                assert err < 2e-9
                cases += 1
    print(f"PASS T3 resolvent: {cases} integrals, max scaled error={worst:.3e} < 2e-9; "
          "r1 derivative tolerance=2e-7; shifted rank-one identities tolerance=2e-10")


run("T3 resolvent", test)

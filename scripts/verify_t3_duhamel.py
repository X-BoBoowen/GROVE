"""Duhamel proposition: positive penalty integrals for heat and powers."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
from scipy.integrate import quad


def diagonal_function(a, v, f):
    vals, vectors = np.linalg.eigh(a)
    return np.sum(vectors[v]**2*f(vals))


def test():
    worst, cases = 0.0, 0
    for a in fixtures():
        v = len(a)//2
        for penalty in [.3, 2.0, np.inf]:
            b = updated(a, v, penalty)
            for kind, param in [("heat", .1), ("heat", 1.3), ("power", .5), ("power", 1), ("power", 2.5)]:
                f = (lambda x: np.exp(-param*x)) if kind == "heat" else (lambda x: x**(-param))
                def positive_integrand(u):
                    au = updated(a, v, u)
                    if kind == "heat":
                        return param*diagonal_function(au, v, f)
                    return param*diagonal_function(au, v, lambda x: x**(-param-1))
                value, error = quad(positive_integrand, 0, penalty,
                                    epsabs=2e-9, epsrel=2e-9, limit=300)
                target = trace_function(a, f)-trace_function(b, f)
                relative = abs(value-target)/max(1, abs(target))
                worst = max(worst, relative)
                assert relative < 2e-7, (kind, param, penalty, value, target, error)
                # Independently check the local trace derivative at finite u.
                u, step = .4, 1e-5
                derivative = (trace_function(updated(a, v, u+step), f)
                              -trace_function(updated(a, v, u-step), f))/(2*step)
                assert abs(derivative+positive_integrand(u)) <= 2e-7*max(1, abs(derivative))
                cases += 1
            # At tau=0 there is a dimension drop only for hard grounding.
            target0 = trace_function(a, lambda x: np.ones_like(x))-trace_function(b, lambda x: np.ones_like(x))
            assert target0 == (1 if penalty == np.inf else 0)
    print(f"PASS T3 Duhamel: {cases} penalty integrals, max scaled error={worst:.3e} < 2e-7; "
          "derivative tolerance=2e-7; tau=0 endpoints exact")


run("T3 Duhamel", test)

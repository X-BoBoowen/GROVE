"""Positive higher-resolvent representation for every s>0."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
from scipy.integrate import quad
from scipy.special import beta as beta_function
import math


def test():
    worst, cases = 0.0, 0
    for a in fixtures():
        v = len(a)//2
        for penalty in [.2, 3, np.inf]:
            b = updated(a, v, penalty)
            for order in range(1, 7):
                for t in [0, .1, 3]:
                    q = scalar_resolvent_drop(a, v, penalty, order, t)
                    ref = trace_function(a+t*np.eye(len(a)), lambda x: x**(-order))
                    ref -= trace_function(b+t*np.eye(len(b)), lambda x: x**(-order))
                    assert q >= -1e-10
                    assert abs(q-ref) < 2e-8*max(1, abs(ref))
            for s in [.15, .5, .95, 1, 1.01, 1.5, 2, 2.8, 5]:
                order = math.floor(s)+1  # Always strictly greater than s.
                def integrand(u):
                    h = (1-u)*a+u*np.eye(len(a))
                    beta_u = np.inf if penalty == np.inf else penalty*(1-u)
                    return scalar_resolvent_drop(h, v, beta_u, order)
                val, _ = quad(integrand, 0, 1, weight="alg", wvar=(order-s-1, s-1),
                              epsabs=2e-9, epsrel=2e-10, limit=300)
                val /= beta_function(s, order-s)
                ref = trace_function(a, lambda x: x**(-s))-trace_function(b, lambda x: x**(-s))
                err = abs(val-ref)/max(1, abs(ref))
                worst = max(worst, err)
                assert err < 2e-8
                if float(s).is_integer():
                    direct = scalar_resolvent_drop(a, v, penalty, int(s), 0)
                    assert abs(val-direct) <= 2e-8*max(1, abs(val))
                cases += 1
    print(f"PASS T3 all positive powers: {cases} integrals, max scaled error={worst:.3e} < 2e-8; "
          "moment/log-series identities tolerance=2e-8; positivity tolerance=1e-10")


run("T3 positive powers", test)

"""Uniform resolvent-curve JL and the positive r2-to-r1 integration route."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))
import math


def test():
    rng = np.random.default_rng(601)
    epsilon, failure, worst = .4, .1, 0.0
    for a in fixtures()[:3]:
        n = len(a)
        lower, upper = np.linalg.eigvalsh(a)[[0, -1]]
        operator_cap = np.sqrt(2*n/failure)
        net_count = 2+math.ceil(4*operator_cap*np.log(upper/lower)/epsilon)
        d = math.ceil(32/epsilon**2*np.log(4*n*net_count/failure))
        q = rng.normal(size=(d, n))/np.sqrt(d)
        assert np.linalg.norm(q, "fro") <= operator_cap
        for shift in np.r_[0, np.geomspace(1e-7, 1e7, 201)]:
            r = np.linalg.inv(a+shift*np.eye(n))
            real = np.sum(r*r, axis=0)
            projected = np.sum((q@r)**2, axis=0)
            error = np.max(abs(projected/real-1))
            worst = max(worst, error)
            assert error < epsilon
            for u in range(n):
                w = r[:, u]/np.linalg.norm(r[:, u])
                derivative = -(r@w-(w@r@w)*w)
                cap = (upper-lower)/(2*(lower+shift)*(upper+shift))
                assert np.linalg.norm(derivative) <= cap+2e-10
        # Integrate via the fixed positive m=2 rule, rather than an unproved
        # finite-grid-to-continuum argument. All added shifts use the same Q.
        for start in [0, .3, 4]:
            m, shifts, weights = power_quadrature(1, lower+start, upper+start, 12)
            assert m == 2
            estimate, numerical_identity = np.zeros(n), np.zeros(n)
            for shift, weight in zip(shifts, weights):
                r = np.linalg.inv(a+(start+shift)*np.eye(n))
                estimate += weight*np.sum((q@r)**2, axis=0)
                numerical_identity += weight*np.sum(r*r, axis=0)
            exact = np.diag(np.linalg.inv(a+start*np.eye(n)))
            assert np.max(abs(numerical_identity/exact-1)) < 1e-8
            assert np.max(abs(estimate/exact-1)) < epsilon+1e-8
    print(f"PASS T6 continuum JL: all-node dense shift grids, max norm-square error={worst:.4f} < {epsilon}; "
          "curve-derivative tolerance=2e-10, positive integral identity tolerance=1e-8")


run("T6 continuum JL", test)

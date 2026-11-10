"""The same two RHS families extend to the higher resolvent moments needed for s>=1."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))


def test():
    rng = np.random.default_rng(603)
    d, worst = 5000, 0.0
    a = fixtures()[0]
    c = np.linalg.cholesky(a).T  # Only a small-test factor; sparse factor is tested separately.
    q = rng.normal(size=(d, len(a)))/np.sqrt(d)
    p = rng.normal(size=(d, len(a)))@c/np.sqrt(d)
    for shift in [0, .5, 10]:
        r = np.linalg.inv(a+shift*np.eye(len(a)))
        y, x = q.copy(), p.copy()
        for ell in range(1, 5):
            y, x = y@r, x@r
            even = np.sum(y*y, axis=0)
            odd = np.sum((x+np.sqrt(shift)*y)**2, axis=0)
            reference_even = np.diag(np.linalg.matrix_power(r, 2*ell))
            reference_odd = np.diag(np.linalg.matrix_power(r, 2*ell-1))
            error = max(np.max(abs(even/reference_even-1)), np.max(abs(odd/reference_odd-1)))
            worst = max(worst, error)
            assert error < .12
            # Deterministic residual recurrence for an inexact repeated solve.
            previous_error, previous = 0.0, q[:3].copy()
            exact = q[:3].copy()
            for _ in range(ell):
                exact = exact@r
                approximate = previous@r+rng.normal(0, 1e-9, previous.shape)
                residual = np.linalg.norm(previous-approximate@(a+shift*np.eye(len(a))), "fro")
                previous_error = (previous_error+residual)/(np.linalg.eigvalsh(a)[0]+shift)
                assert np.linalg.norm(approximate-exact, "fro") <= previous_error+1e-11
                previous = approximate
    print(f"PASS T6 higher moments: orders 1..8, all nodes at three shifts; max relative sketch error={worst:.4f} < .12; "
          "residual recurrence tolerance=1e-11")


run("T6 higher moments", test)

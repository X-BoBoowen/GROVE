"""Shared Lanczos theorem: scalar moments, residual bounds, exact termination."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def lanczos(a, v, depth):
    n = len(a)
    q = np.eye(n)[:, v]
    previous = np.zeros(n)
    alpha, offdiag, vectors = [], [], []
    beta_previous = 0.0
    gamma, next_vector = 0.0, np.zeros(n)
    for j in range(depth):
        vectors.append(q.copy())
        w = a@q-beta_previous*previous
        value = q@w
        alpha.append(value)
        w -= value*q
        # Full reorthogonalization keeps this small numerical test reliable.
        for _ in range(2):
            for basis in vectors:
                w -= (basis@w)*basis
        gamma = np.linalg.norm(w)
        if gamma < 1e-12:
            gamma, next_vector = 0.0, np.zeros(n)
            break
        next_vector = w/gamma
        if j == depth-1:
            break
        offdiag.append(gamma)
        previous, q, beta_previous = q, next_vector, gamma
    t = np.diag(alpha)
    if offdiag:
        t += np.diag(offdiag, 1)+np.diag(offdiag, -1)
    return np.column_stack(vectors), t, gamma, next_vector


def test():
    cases, worst_exact, worst_recurrence, truncated_difference = 0, 0.0, 0.0, 0.0
    matrices = fixtures()+[np.diag([1.0, 2, 4]), np.array([[2., -1], [-1, 2]])]
    for a in matrices:
        n, v = len(a), 0
        amin = np.linalg.eigvalsh(a)[0]
        e = np.eye(n)[:, v]
        for depth in sorted(set([1, min(2, n), min(4, n), n])):
            q, t, gamma, next_vector = lanczos(a, v, depth)
            h = len(t)
            residual = a@q-q@t
            residual[:, -1] -= gamma*next_vector
            recurrence_error = np.linalg.norm(residual)
            worst_recurrence = max(worst_recurrence, recurrence_error)
            assert recurrence_error < 1e-10
            assert np.linalg.norm(q.T@q-np.eye(h)) < 1e-10
            assert np.linalg.norm(q[:, 0]-e) < 1e-12
            for shift in [0, 1e-4, .1, 1, 10, 1e4]:
                z = np.linalg.solve(t+shift*np.eye(h), np.eye(h)[:, 0])
                y = q@z
                x = np.linalg.solve(a+shift*np.eye(n), e)
                r1h, r2h = z[0], z@z
                r1, r2 = x[v], x@x
                rho = gamma*abs(z[-1])
                assert abs(np.linalg.norm(e-(a+shift*np.eye(n))@y)-rho) < 1e-10
                tol = 2e-10*max(1, r1, r2)
                assert -tol <= r1-r1h <= rho**2/(amin+shift)+tol
                epsilon = rho/(amin+shift)
                assert max(0, np.sqrt(r2h)-epsilon)**2-tol <= r2
                assert r2 <= (np.sqrt(r2h)+epsilon)**2+tol
                if gamma == 0:
                    for order in range(1, 8):
                        exact = diagonal_moment(a, v, order, shift)
                        projected = diagonal_moment(t, 0, order, shift)
                        err = abs(exact-projected)/max(1, abs(exact))
                        worst_exact = max(worst_exact, err)
                        assert err < 2e-8
                else:
                    truncated_difference = max(truncated_difference, abs(r1-r1h), abs(r2-r2h))
                cases += 1
            for penalty in [.3, 2, np.inf]:
                for f in [lambda x: x**(-.25), lambda x: 1/x,
                          lambda x: x**(-2.5), lambda x: np.exp(-.7*x)]:
                    reduced = trace_function(t, f)-trace_function(updated(t, 0, penalty), f)
                    assert reduced >= -1e-10
                    if gamma == 0:
                        exact = trace_function(a, f)-trace_function(updated(a, v, penalty), f)
                        assert abs(exact-reduced) < 2e-8*max(1, abs(exact))
    assert truncated_difference > 1e-3
    print(f"PASS T3 Lanczos: {cases} shifted solves; max recurrence error={worst_recurrence:.3e} < 1e-10; "
          f"terminal moment error={worst_exact:.3e} < 2e-8; residual interval tolerance=2e-10; "
          f"nonzero truncation discrepancy={truncated_difference:.3e} (not treated as exact)")


def diagonal_moment(a, v, order, shift):
    vals, vec = np.linalg.eigh(a)
    return np.sum(vec[v]**2/(vals+shift)**order)


run("T3 Lanczos", test)

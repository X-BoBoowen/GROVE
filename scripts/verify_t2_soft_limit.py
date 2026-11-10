"""Lemma 6: PSD inverse block identity and uniform finite-beta error bound."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def test():
    edges = [(u, v, 1) for u in range(3) for v in range(3, 6)]
    l, n = laplacian(6, edges), 6
    cases, max_error, max_ratio = 0, 0.0, 0.0
    for selected in subsets(n):
        if not selected or len(selected) == n:
            continue
        u, v = sorted(set(range(n))-selected), sorted(selected)
        order = u+v
        for sigma in [0, 1]:
            a = (l+sigma*np.eye(n))[np.ix_(order, order)]
            r = len(u)
            c, d, e = a[:r, :r], a[:r, r:], a[r:, r:]
            ci = np.linalg.inv(c)
            w = np.vstack([-ci@d, np.eye(len(v))])
            j = np.zeros((n, n))
            j[:r, :r] = ci
            for beta in [1, 10, 1e3, 1e5]:
                f = e + beta*np.eye(len(v)) - d.T@ci@d
                correction = w@np.linalg.solve(f, w.T)
                inverse = np.linalg.inv(a+np.diag([0]*r+[beta]*len(v)))
                err = np.linalg.norm(inverse-j-correction, 2)
                max_error = max(max_error, err/max(1, np.linalg.norm(inverse, 2)))
                assert max_error < 2e-10
                assert np.linalg.eigvalsh(correction)[0] > -1e-10
                epsilon = (1+np.linalg.norm(ci@d, 2)**2)/beta
                assert np.linalg.norm(correction, 2) <= epsilon+1e-10
                assert epsilon <= (1+9*n**4)/beta+1e-10
                for s in [.1, .25, .5, .9]:
                    actual = np.sum(np.linalg.eigvalsh(inverse)**s)-np.sum(np.linalg.eigvalsh(ci)**s)
                    bound = n*epsilon**s
                    assert -2e-9 <= actual <= bound+2e-9
                    max_ratio = max(max_ratio, actual/bound)
                    cases += 1
    print(f"PASS T2 soft limit: {cases} checks; block relative error={max_error:.3e} < 2e-10; "
          f"max actual/bound={max_ratio:.4f}; trace tolerance=2e-9")


run("T2 soft limit", test)

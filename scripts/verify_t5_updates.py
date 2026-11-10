"""Exact soft/hard r1 and r2 cache updates, including the R^2 column."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def test():
    worst, count = 0.0, 0
    for a in fixtures():
        for shift in [0, .1, 3, 100]:
            r = np.linalg.inv(a+shift*np.eye(len(a)))
            for v in range(len(a)):
                x, y = r[:, v], (r@r)[:, v]
                for beta in [.1, 2, np.inf]:
                    c = 1/((0 if beta == np.inf else 1/beta)+x[v])
                    first = np.diag(r)-c*x*x
                    second = np.diag(r@r)-2*c*x*y+c*c*(x@x)*x*x
                    b = updated(a, v, beta)
                    actual = np.linalg.inv(b+shift*np.eye(len(b))) if len(b) else b
                    keep = [u for u in range(len(a)) if beta != np.inf or u != v]
                    error = max(np.max(abs(first[keep]-np.diag(actual)), initial=0),
                                np.max(abs(second[keep]-np.diag(actual@actual)), initial=0))
                    scale = max(1, np.max(np.diag(r@r)))
                    worst = max(worst, error/scale)
                    assert error <= 3e-10*scale
                    if beta == np.inf:
                        assert abs(first[v])+abs(second[v]) <= 3e-10*scale
                    count += 1
    print(f"PASS T5 exact updates: {count} node/shift/penalty cases; max scaled error={worst:.3e} < 3e-10")


run("T5 updates", test)

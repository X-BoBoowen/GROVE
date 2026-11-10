"""Strict-convexity hardness gap: all positive powers and positive-time heat."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def test():
    cases = 0
    graphs = [(4, [(i, j, 1) for i in range(4) for j in range(i+1, 4)]),
              (6, [(i, j, 1) for i in range(3) for j in range(3, 6)])]
    for n, edges in graphs:
        for sigma in [0, 1]:
            a = laplacian(n, edges)+sigma*np.eye(n)
            funcs = [(lambda x, s=s: x**(-s), s*(s+1)/(6+sigma)**(s+2))
                     for s in [.2, 1, 2, 5]]
            funcs += [(lambda x, t=t: np.exp(-t*x), t*t*np.exp(-t*(6+sigma)))
                      for t in [.1, .7, 2]]
            for f, gap in funcs:
                vals = {}
                for selected in subsets(n):
                    if not selected or len(selected) == n:
                        continue
                    inside = sum(u not in selected and v not in selected for u, v, _ in edges)
                    value = trace_function(grounded(a, selected, np.inf), f)
                    base = (n-len(selected))*f(3+sigma)
                    assert value-base >= gap*inside-1e-9*max(1, value)
                    if inside == 0:
                        assert abs(value-base) < 1e-10
                    else:
                        assert value > base+gap*.99
                    vals[selected] = value
                    cases += 1
                for k in range(1, n):
                    eligible = [z for z in vals if len(z) == k]
                    cover_exists = any(all(u in z or v in z for u, v, _ in edges) for z in eligible)
                    best = min(eligible, key=vals.get)
                    assert cover_exists == all(u in best or v in best for u, v, _ in edges)
    print(f"PASS T2 extended: {cases} gap checks and all cardinality reductions; tolerance=1e-9")


run("T2 extended", test)

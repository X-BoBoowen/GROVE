"""Lemma 5: cubic vertex-cover equality and a quantitative fractional trace gap."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def cubic_graphs():
    k4 = [(u, v, 1) for u in range(4) for v in range(u+1, 4)]
    k33 = [(u, v, 1) for u in range(3) for v in range(3, 6)]
    prism = [(i, (i+1)%3, 1) for i in range(3)] + [(i+3, (i+1)%3+3, 1) for i in range(3)]
    prism += [(i, i+3, 1) for i in range(3)]
    return [(4, k4), (6, k33), (6, prism)]


def test():
    cases, min_ratio = 0, np.inf
    for n, edges in cubic_graphs():
        l = laplacian(n, edges)
        assert np.all(np.diag(l) == 3)
        for sigma in [0, 1, 2]:
            for s in [.1, .25, .5, .9, 1, 2]:
                gap = s*(s+1)/(6+sigma)**(s+2)
                for selected in subsets(n):
                    if not selected or len(selected) == n:
                        continue
                    inside = sum(u not in selected and v not in selected for u, v, _ in edges)
                    b = grounded(l+sigma*np.eye(n), selected, np.inf)
                    val = trace_function(b, lambda x: x**(-s))
                    base = (n-len(selected)) / (3+sigma)**s
                    assert val-base >= gap*inside-2e-10
                    if inside == 0:
                        assert abs(val-base) < 2e-10
                    else:
                        assert val > base + .99*gap
                        min_ratio = min(min_ratio, (val-base)/(gap*inside))
                    if sigma == 0:
                        assert np.linalg.eigvalsh(b)[0] >= 1/n**2 - 1e-12
                    cases += 1
    print(f"PASS T2 cubic-cover gap: {cases} subsets/powers/shifts; "
          f"minimum observed excess / proved gap={min_ratio:.4f}; tolerance=2e-10")


if __name__ == "__main__":
    run("T2 cubic-cover gap", test)

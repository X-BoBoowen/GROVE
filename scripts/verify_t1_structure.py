"""Theorem 2: exhaustive edge cases, CM atoms, strictness and hard-limit offset."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def test():
    graphs = [laplacian(4, [(0, 1, .25), (1, 2, 2), (2, 3, .7), (0, 2, 1.3)]),
              laplacian(4, [(0, 1, .25), (2, 3, 2)])]
    funcs = [("zero", lambda x: 0 * x), ("constant", lambda x: 0 * x + 2),
             ("atom+continuous", lambda x: 2 + x ** (-.37) + .8 * np.exp(-1.2*x)),
             ("atoms", lambda x: .2*np.exp(-.01*x) + 3*np.exp(-2*x))]
    count, worst = 0, 0.0
    for gi, l in enumerate(graphs):
        for sigma in ([0, .13] if gi == 0 else [.13]):
            seed = {0} if sigma == 0 else set()
            a = l + sigma*np.eye(4)
            for beta in [.03, 2, 30, np.inf]:
                for name, f in funcs:
                    values = {s: trace_function(grounded(a, s, beta), f)
                              for s in subsets(4) if seed <= s}
                    _, mixed = check_set_function(values, 4, 3e-10)
                    worst = min(worst, mixed)
                    for s, val in values.items():
                        for v in set(range(4)) - s:
                            drop = val - values[s | {v}]
                            strict = name != "zero" and (beta == np.inf or name != "constant")
                            if strict:
                                assert drop > 1e-10, (gi, sigma, beta, name, s, v, drop)
                            else:
                                assert abs(drop) < 1e-12
                    count += 1
    # General CM functions have a possible atom at t=0: keep its dimension term.
    a, s = graphs[0] + np.eye(4), {0, 2}
    f = lambda x: 2 + 1/x
    target = trace_function(grounded(a, s, np.inf), f) + 2*len(s)
    errors = [abs(trace_function(grounded(a, s, b), f) - target) for b in [1e3, 1e4, 1e5]]
    assert errors[2] < errors[1] < errors[0] and errors[-1] < 5e-5
    print(f"PASS T1 structure: {count} exhaustive objectives, mixed min={worst:.3e}, "
          "relative tolerance=3e-10; strictness threshold=1e-10; "
          f"CM hard-limit offset error={errors[-1]:.3e} < 5e-5")


run("T1 structure", test)

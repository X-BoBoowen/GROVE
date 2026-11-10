"""Numerical checks of the mathematical distinctions used in the positioning paragraph.

Literature attribution is supported by the cited papers, not by this script.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))


def test():
    # CFCC is exactly the reciprocal of the hard s=1 trace, up to factor n.
    l = laplacian(5, [(0, 1, .7), (1, 2, 1), (2, 3, 1.5), (3, 4, .8), (0, 4, .4)])
    a = grounded(l, {0, 2}, np.inf)
    source_voltage_sum = sum(np.linalg.solve(a, np.eye(len(a))[:, u])[u] for u in range(len(a)))
    assert abs(5/source_voltage_sum-5/trace_function(a, lambda x: 1/x)) < 1e-11
    # Provided FastFrac/POLAR contrast: edge addition has different structure.
    edges = [(0, 1, 1), (1, 2, 1), (0, 4, 1), (4, 3, 1)]
    additions = [[], [(0, 2, 1)], [(0, 3, 1)], [(0, 2, 1), (0, 3, 1)]]
    for exponent in [.5, -.5, -1]:
        traces = [trace_function(np.eye(5)+10*laplacian(5, edges+extra), lambda x: x**exponent)
                  for extra in additions]
        mixed = traces[0]-traces[1]-traces[2]+traces[3]
        assert mixed < -1e-6
    # Zhou's seven-node path: lambda_min fails diminishing gains, whereas
    # the CM trace still has diminishing drops for precisely the same sets.
    l = laplacian(7, [(i, i+1, 1) for i in range(6)])
    small, large, v = {0}, {0, 1}, 5
    matrices = [grounded(l, z, np.inf) for z in [small, small|{v}, large, large|{v}]]
    lam = [np.linalg.eigvalsh(a)[0] for a in matrices]
    assert lam[1]-lam[0] < lam[3]-lam[2]-.1
    for f in [lambda x: 1/x, lambda x: x**(-2), lambda x: x**(-4), lambda x: np.exp(-.7*x)]:
        phi = [trace_function(a, f) for a in matrices]
        assert phi[0]-phi[1] >= phi[2]-phi[3]-1e-9*max(phi)
    print("PASS T8 positioning checks: CFCC identity tolerance=1e-11; positive/negative edge-trace contrasts; "
          "Zhou path eigenvalue control and CM diminishing drops tolerance=1e-9")


run("T8 positioning checks", test)

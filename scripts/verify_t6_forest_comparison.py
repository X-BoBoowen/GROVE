"""Exact enumeration of weighted rooted forests and the direct counting variance."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))


def forest_probabilities(n, edges, killing):
    total, root_matrix = 0.0, np.zeros((n, n))
    for edge_mask in range(1 << len(edges)):
        parent, product = list(range(n)), 1.0
        def find(v):
            while parent[v] != v:
                v = parent[v]
            return v
        valid = True
        for j, (u, v, weight) in enumerate(edges):
            if edge_mask >> j & 1:
                pu, pv = find(u), find(v)
                if pu == pv:
                    valid = False
                    break
                parent[pu] = pv
                product *= weight
        if not valid:
            continue
        components = {}
        for v in range(n):
            components.setdefault(find(v), []).append(v)
        components = list(components.values())
        for roots in itertools.product(*components):
            weight = product*killing**len(components)
            total += weight
            for component, root in zip(components, roots):
                root_matrix[component, root] += weight
    return total, root_matrix/total


def test():
    edges = [(0, 1, .7), (1, 2, 1.3), (2, 3, .5), (0, 3, 1.1), (0, 2, .4)]
    l, worst = laplacian(4, edges), 0.0
    for killing in [.01, .4, 3]:
        total, probabilities = forest_probabilities(4, edges, killing)
        a = l+killing*np.eye(4)
        r = np.linalg.inv(a)
        error = np.max(abs(probabilities-killing*r))
        worst = max(worst, error)
        assert error < 2e-10
        assert abs(total/np.linalg.det(a)-1) < 2e-10
        assert np.max(abs(np.sum(probabilities**2, axis=1)-killing**2*np.diag(r@r))) < 2e-10
    for n in [4, 16, 64]:
        killing = .01
        probability1 = 1/n+(1-1/n)*killing/(n+killing)
        probability2 = 1/n+(1-1/n)*(killing/(n+killing))**2
        for probability in [probability1, probability2]:
            assert 1/n <= probability <= 2/n
            # N samples have exact relative mean-square error (1-p)/(N*p).
            required_for_rms_point_one = (1-probability)/(probability*.1**2)
            assert required_for_rms_point_one >= (n/2-1)/.1**2
    print(f"PASS T6 forest comparison: exact weighted-forest enumeration error={worst:.3e} < 2e-10; "
          "root/collision identities and complete-graph counting variance bounds pass")


run("T6 forest comparison", test)

"""Numerical helpers for the T7--T9 proof checks."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase2_utils.py").read_text(), "_phase2_utils.py", "exec"))


def moments(a, v, order, shift=0):
    vals, vectors = np.linalg.eigh(a)
    z, weights = 1/(vals+shift), vectors[v]**2
    return np.array([np.sum(weights*z**j) for j in range(1, order+1)])


def moment_marginal(rs, penalty, order):
    c = 1/((0 if penalty == np.inf else 1/penalty)+rs[0])
    coeff = np.r_[1.0, c*rs[1:order+1]]
    log_coeff = np.zeros(order+1)
    for k in range(1, order+1):
        log_coeff[k] = (k*coeff[k]-sum(j*log_coeff[j]*coeff[k-j] for j in range(1, k)))/k
    return order*log_coeff[order]


def positive_integer_marginal(a, v, penalty, order):
    r = np.linalg.inv(a)
    x = r[:, v]
    c = 1/((0 if penalty == np.inf else 1/penalty)+r[v, v])
    k = r-c*np.outer(x, x)
    eig, vec = np.linalg.eigh(k)
    assert eig[0] >= -1e-10*max(1, np.linalg.norm(r, 2))
    eig = np.maximum(eig, 0)  # Remove only roundoff at the hard zero eigenvalue.
    mass = (vec.T@x)**2
    d = np.r_[0.0, [c*np.sum(mass*eig**j) for j in range(order)]]
    h = np.r_[1.0, np.zeros(order)]
    for j in range(1, order+1):
        h[j] = sum(d[i]*h[j-i] for i in range(1, j+1))
    value = sum(j*d[j]*h[order-j] for j in range(1, order+1))
    return value, d, h


def moment_error_bound(order, epsilon):
    return 2**order*np.expm1(-order*np.log1p(-epsilon))


def required_moment_precision(order, marginal_error):
    return -np.expm1(-np.log1p(marginal_error/2**order)/order)


def family_functions(power=None, heat=None):
    if power is not None:
        return (lambda x: x**(-power)), (lambda x: power*x**(-power-1))
    return (lambda x: np.exp(-heat*x)), (lambda x: heat*np.exp(-heat*x))


def tridiagonal_moments(t, count):
    from scipy.linalg import solve_banded
    h = len(t)
    band = np.zeros((3, h))
    band[1] = np.diag(t)
    if h > 1:
        band[0, 1:] = np.diag(t, 1)
        band[2, :-1] = np.diag(t, -1)
    vector = np.eye(h)[:, 0]
    result = []
    for _ in range(count):
        vector = solve_banded((1, 1), band, vector)
        result.append(vector[0])
    return np.array(result)


def certify_marginal(a, v, penalty, theta, lower, upper, power=None, heat=None):
    """One incremental run; small dense reference orthogonalization is for tests."""
    f, g = family_functions(power, heat)
    n = len(a)
    vector, old, previous_beta = np.eye(n)[:, v], np.zeros(n), 0.0
    basis, diagonal, offdiag = [], [], []
    checkpoints = 0
    for step in range(1, n+1):
        basis.append(vector.copy())
        residual = a@vector-previous_beta*old
        alpha = vector@residual
        diagonal.append(alpha)
        residual -= alpha*vector
        for _ in range(2):
            for q in basis:
                residual -= (q@residual)*q
        gamma = np.linalg.norm(residual)
        terminal = gamma < 1e-12 or step == n
        if terminal or step & (step-1) == 0:
            checkpoints += 1
            t = np.diag(diagonal)
            if offdiag:
                t += np.diag(offdiag, 1)+np.diag(offdiag, -1)
            if power is not None and float(power).is_integer():
                p = int(power)
                lo = moment_marginal(tridiagonal_moments(t, p+1), penalty, p)
            else:
                lo = trace_function(t, f)-trace_function(updated(t, 0, penalty), f)
            assert lo >= -1e-10
            lo = max(lo, 0.0)
            if terminal:
                hi = lo
            else:
                end = upper+(0 if penalty == np.inf else penalty)
                rho = (np.sqrt(2*end-lower)+np.sqrt(lower))/(np.sqrt(2*end-lower)-np.sqrt(lower))
                degree = 2*step-2 if penalty == np.inf else 2*step-1
                poly_error = 2*g(lower/2)*rho**(-degree)/(rho-1)
                width = (upper-lower)*poly_error if penalty == np.inf else 2*penalty*poly_error
                hi = lo+width
            if lo >= (1-theta)*hi:
                return lo, hi, step, checkpoints
        old, vector, previous_beta = vector, residual/gamma, gamma
        offdiag.append(gamma)
    raise AssertionError("Invariant termination must certify the marginal")


def coarse_jl_keys(a, penalty, order, rng, epsilon=.4, failure=.1):
    """Actual Gaussian initial sketches; uses dense solves only as a small-test reference."""
    n = len(a)
    d = int(np.ceil(128/epsilon**2*np.log(12*n*(order+1)/failure)))
    c = np.linalg.cholesky(a).T
    q = rng.normal(size=(d, n))/np.sqrt(d)
    p = rng.normal(size=(d, n))@c/np.sqrt(d)
    r = np.linalg.inv(a)
    x, y, first, last = p.copy(), q.copy(), None, None
    depth = (order+2)//2
    for ell in range(1, depth+1):
        x, y = x@r, y@r
        if ell == 1:
            first = np.sum(x*x, axis=0)
        if 2*ell == order+1:
            last = np.sum(y*y, axis=0)
        if 2*ell-1 == order+1:
            last = np.sum(x*x, axis=0)
    true_first = np.diag(r)
    true_last = np.diag(np.linalg.matrix_power(r, order+1))
    assert np.max(abs(first/true_first-1)) < epsilon
    assert np.max(abs(last/true_last-1)) < epsilon
    offset = 0 if penalty == np.inf else 1/penalty
    keys = order*(last/(1-epsilon))/(offset+first/(1+epsilon))
    return keys, 2*d*depth


def certified_celf(a, penalty, budget, theta, initial_upper, power=None, heat=None):
    import heapq
    n = len(a)
    lower = .5*np.linalg.eigvalsh(a)[0]
    upper = np.linalg.eigvalsh(a)[-1]+(0 if penalty == np.inf else penalty)
    keys = np.array(initial_upper, dtype=float)
    heap = [(-keys[u], u) for u in range(n)]
    heapq.heapify(heap)
    selected, stamp, lower_values = set(), np.full(n, -1), np.zeros(n)
    refreshes, selections, pops = [], [], 0
    for round_index in range(budget):
        current = grounded(a, selected, penalty)
        keep = [u for u in range(n) if u not in selected]
        # A snapshot is returned only for testing the invariant; it is not used to select.
        snapshot = {u: keys[u] for u in keep}
        while True:
            _, u = heapq.heappop(heap)
            pops += 1
            if stamp[u] != round_index:
                local_u = keep.index(u) if penalty == np.inf else u
                lo, hi, depth, checkpoints = certify_marginal(
                    current, local_u, penalty, theta, lower, upper, power, heat)
                keys[u] = min(keys[u], hi)
                lower_values[u], stamp[u] = lo, round_index
                refreshes.append((round_index, u, depth, checkpoints))
            next_upper = -heap[0][0] if heap else 0.0
            if lower_values[u] >= (1-theta)*next_upper:
                selections.append((frozenset(selected), u, lower_values[u], next_upper, snapshot))
                selected.add(u)
                break
            heapq.heappush(heap, (-keys[u], u))
    return frozenset(selected), refreshes, selections, pops

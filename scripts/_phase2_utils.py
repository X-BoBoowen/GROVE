"""Numerical helpers for T4--T6; no production algorithm is installed here."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
from scipy.special import roots_jacobi, roots_legendre, beta as beta_function


def lanczos(a, v, depth):
    q = np.eye(len(a))[:, v]
    vectors, diagonal, off = [], [], []
    old, previous_beta = np.zeros(len(a)), 0.0
    for j in range(depth):
        vectors.append(q.copy())
        w = a@q-previous_beta*old
        diagonal.append(q@w)
        w -= diagonal[-1]*q
        for _ in range(2):
            for basis in vectors:
                w -= (basis@w)*basis
        gamma = np.linalg.norm(w)
        if gamma < 1e-12:
            gamma = 0.0
            break
        if j+1 == depth:
            break
        old, q, previous_beta = q, w/gamma, gamma
        off.append(gamma)
    t = np.diag(diagonal)
    if off:
        t += np.diag(off, 1)+np.diag(off, -1)
    return np.column_stack(vectors), t, gamma


def diagonal_function(a, f):
    vals, vec = np.linalg.eigh(a)
    return np.sum(vec[0]**2*f(vals))


def radau(t, gamma, lower, penalty=0.0):
    j = t.copy()
    j[0, 0] += penalty
    if gamma == 0:
        return j
    h = len(t)
    out = np.zeros((h+1, h+1))
    out[:h, :h] = j
    out[h-1, h] = out[h, h-1] = gamma
    out[h, h] = lower+gamma**2*np.linalg.inv(j-lower*np.eye(h))[-1, -1]
    return out


def power_quadrature(s, lower, upper, order):
    """Positive composite rule for x^-s, using shifted inverse powers m>s."""
    m = int(np.floor(s))+1
    shift, weight = [], []
    # [0,a]: Jacobi weight z^(m-s-1).
    z, w = roots_jacobi(order, 0, m-s-1)
    z, w = (z+1)/2, w/2**(m-s)
    shift.extend(lower*z)
    weight.extend(lower**(m-s)*w)
    # Geometric interior panels: ordinary Gauss--Legendre.
    left = lower
    z, w = roots_legendre(order)
    while left < upper*(1-1e-14):
        right = min(2*left, upper)
        nodes = (left+right)/2+(right-left)/2*z
        shift.extend(nodes)
        weight.extend((right-left)/2*w*nodes**(m-s-1))
        left = right
    # [B,infinity): z=B/t, Jacobi weight z^(s-1).
    z, w = roots_jacobi(order, 0, s-1)
    z, w = (z+1)/2, w/2**s
    shift.extend(upper/z)
    weight.extend(upper**(m-s)*w/z**m)
    return m, np.array(shift), np.array(weight)/beta_function(s, m-s)


def quadrature_values(x, m, shift, weight):
    x = np.asarray(x)
    f = np.sum(weight[:, None]/(shift[:, None]+x.ravel()[None, :])**m, axis=0)
    g = m*np.sum(weight[:, None]/(shift[:, None]+x.ravel()[None, :])**(m+1), axis=0)
    return f.reshape(x.shape), g.reshape(x.shape)

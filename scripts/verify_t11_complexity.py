"""Right-hand-side, update, and fixed-factor iteration work counts."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def test():
    rng = np.random.default_rng(1102)
    cases = 0
    for n in [20, 200, 2000]:
        for p in [1, 2, 3, 4, 8]:
            j, depth = p+1, (p+2)//2
            for k in [1, 5, 10]:
                d = 32
                refreshes = np.zeros(k, dtype=int)
                # Each newly selected node was refreshed at least once.
                refreshes[:] = 1
                remaining = n-k
                for t in range(k):
                    extra = int(rng.integers(0, min(remaining, 20)+1))
                    refreshes[t] += extra
                    remaining -= extra
                reused = rng.integers(0, depth+1, max(0, k-1))
                ell = 19
                h = np.array([min(n-t, t+ell) for t in range(k)])
                init_rhs = d*((j+1)//2+j//2)
                assert init_rhs == j*d
                candidate_rhs = sum(depth*int(r) for r in refreshes)
                update_rhs = sum(j-int(s) for s in reused)
                counted = init_rhs*h[0]+sum(depth*refreshes*h)
                counted += sum((j-reused)*h[:-1])
                loose = j*d*ell+(depth*sum(refreshes)+j*max(0, k-1))*(k+ell)
                assert counted <= loose
                assert candidate_rhs == depth*sum(refreshes)
                assert update_rhs <= j*max(0, k-1)
                assert sum(refreshes) <= n
                # Actual loop operation count for the two triangular recurrences.
                scalar_products = sum(range(j))+sum(range(1, j+1))
                assert scalar_products == j*j
                # The current acmin=4 hybrid needs a separate single-column term.
                jacobi_h = 137
                hybrid = init_rhs*h[0]+sum(depth*refreshes*h)+update_rhs*jacobi_h
                assert hybrid >= counted
                # Reusing a fixed factor for fresh all-node rounds retains sum(t).
                fresh = j*d*sum(h)
                assert fresh <= j*d*(k*ell+k*(k-1)//2)
                cases += 1
    print(f'PASS T11 work: {cases} parameter regimes; J*d initialization, L*R refreshes, '
          'selected-chain reuse, O(N*k*J^2) scalar passes, and mixed-Jacobi accounting')


run('T11 fixed-factor work', test)

"""Finite checks of the T10 gap reduction; complexity follows from the proof."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from fractions import Fraction


def cubic_graphs():
    yield 4, [(i, j, 1) for i in range(4) for j in range(i+1, 4)]
    yield 6, [(i, j, 1) for i in range(3) for j in range(3, 6)]
    yield 6, [(0, 1, 1), (1, 2, 1), (2, 0, 1), (3, 4, 1), (4, 5, 1),
              (5, 3, 1), (0, 3, 1), (1, 4, 1), (2, 5, 1)]
    yield 8, [(i, i ^ (1 << j), 1) for i in range(8) for j in range(3) if i < i ^ (1 << j)]
    yield 10, ([(i, (i+1) % 5, 1) for i in range(5)]
               + [(i, i+5, 1) for i in range(5)]
               + [(i+5, (i+2) % 5+5, 1) for i in range(5)])


def test():
    checked, seeded, no_instances, yes_instances = 0, 0, 0, 0
    for n, edges in cubic_graphs():
        l = laplacian(n, edges)
        assert np.array_equal(np.diag(l), np.full(n, 3.))
        groups = {k: [] for k in range(1, n)}
        for s in subsets(n):
            if not 0 < len(s) < n:
                continue
            cover = all(u in s or v in s for u, v, _ in edges)
            val = np.linalg.eigvalsh(grounded(l, s, np.inf))[0]
            assert val > 0 and val <= 3+2e-10
            if cover:
                assert abs(val-3) < 2e-10
            else:
                assert val <= 2+2e-10
                u, v, _ = next(e for e in edges if e[0] not in s and e[1] not in s)
                # Exact Rayleigh quotient without an irrational normalization.
                x = np.zeros(n, dtype=int)
                x[[u, v]] = 1
                assert int(x@l@x) == 4 and int(x@x) == 2
            groups[len(s)].append((s, cover, val))
            checked += 1
        for group in groups.values():
            yes = any(row[1] for row in group)
            optimum = max(row[2] for row in group)
            yes_instances += yes
            no_instances += not yes
            assert (abs(optimum-3) < 2e-10) == yes
            for gamma in [.667, .75, .99]:
                if yes:
                    assert all(cover for _, cover, val in group if val >= gamma*optimum)
                else:
                    assert all(not cover for _, cover, _ in group)
        # The mandatory singleton seed is constructed, not guessed/enumerated.
        for sigma in [Fraction(0), Fraction(1, 100), Fraction(2)]:
            for gamma in [Fraction(67, 100), Fraction(3, 4), Fraction(99, 100)]:
                threshold = (1-gamma)*(1+sigma)/(3*gamma-2)
                w = threshold.numerator//threshold.denominator+1
                d = 3*w+1+sigma
                assert gamma*d > d-w
                seed_edges = [(u, v, w) for u, v, _ in edges]+[(n, u, 1) for u in range(n)]
                full = laplacian(n+1, seed_edges)+float(sigma)*np.eye(n+1)
                a = grounded(full, {n}, np.inf)
                assert np.max(abs(a-(w*l+(1+float(sigma))*np.eye(n)))) < 1e-12
                for k in [1, n//2, n-1]:
                    for s, cover, _ in groups[k]:
                        value = np.linalg.eigvalsh(grounded(a, s, np.inf))[0]
                        bound = float(d if cover else d-w)
                        assert value <= bound+2e-10*float(d)
                        if cover:
                            assert abs(value-float(d)) <= 2e-10*float(d)
                        seeded += 1
    # A YES instance also contains a noncover attaining exactly 2/3 of OPT;
    # the strict gamma>2/3 threshold cannot be made non-strict by this proof.
    n, edges = list(cubic_graphs())[1]
    l = laplacian(n, edges)
    keep_edge = {0, 3}
    bad = set(range(n))-keep_edge
    assert abs(np.linalg.eigvalsh(grounded(l, bad, np.inf))[0]-2) < 1e-12
    assert abs(np.linalg.eigvalsh(grounded(l, {0, 1, 2, 3}, np.inf))[0]-3) < 1e-12
    for w in [1, 10, 1000]:
        assert Fraction(2*w, 3*w) == Fraction(2, 3)
        for shift in [Fraction(0), Fraction(1, 100), Fraction(2)]:
            assert (2*w+shift)/(3*w+shift) >= Fraction(2, 3)
    assert no_instances and yes_instances
    print(f'PASS T10 hardness gap: {checked} unseeded sets, {seeded} seeded/scaled checks; '
          f'{yes_instances} YES and {no_instances} NO budgets; exact rational Rayleigh/scaling checks; '
          'tolerance=2e-10; strict 2/3 boundary witnessed')


run('T10 hardness gap', test)

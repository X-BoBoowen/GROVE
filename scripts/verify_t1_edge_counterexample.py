"""Proposition 4: exact rational edge-addition counterexample (expected control)."""
from fractions import Fraction as F


def inverse(a):
    n = len(a)
    aug = [list(row) + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if aug[i][j])
        aug[j], aug[pivot] = aug[pivot], aug[j]
        scale = aug[j][j]
        aug[j] = [x / scale for x in aug[j]]
        for i in range(n):
            if i != j:
                scale = aug[i][j]
                aug[i] = [x - scale*y for x, y in zip(aug[i], aug[j])]
    return [row[n:] for row in aug]


def test():
    base = [(0, 1), (1, 2), (0, 4), (4, 3)]
    values = []
    for extra in [[], [(0, 2)], [(0, 3)], [(0, 2), (0, 3)]]:
        a = [[F(i == j, 10) for j in range(5)] for i in range(5)]
        for u, v in base + extra:
            a[u][u] += 1
            a[v][v] += 1
            a[u][v] -= 1
            a[v][u] -= 1
        r = inverse(a)
        assert all(sum(a[i][k]*r[k][j] for k in range(5)) == (i == j)
                   for i in range(5) for j in range(5))
        values.append(sum(r[i][i] for i in range(5)))
    mixed = values[0] - values[1] - values[2] + values[3]
    assert mixed < 0, mixed
    print(f"PASS T1 edge control: H={list(map(str, values))}; "
          f"mixed={mixed} = {float(mixed):.12g} < 0; tolerance=0 (exact rational)")


try:
    test()
except Exception as exc:
    print(f"FAIL T1 edge control: {exc}")
    raise

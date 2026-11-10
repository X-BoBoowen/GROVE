"""Theorem 7: finite exhaustive reduction tests, including beta/input scaling.

These tests check the reduction identities and separation, not NP-hardness
itself; the latter follows from the reduction proof in notes/theory.tex.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name("_verify_utils.py").read_text(), "_verify_utils.py", "exec"))
import math
from fractions import Fraction


def test():
    n = 6
    edges = [(u, v, 1) for u in range(3) for v in range(3, 6)]
    l = laplacian(n, edges)
    for s in [.25, .5, .75]:
        delta = s*(s+1)/6**(s+2)
        p = math.ceil(1/s)+1
        constant = math.ceil((8/delta)**(1/s))
        beta = constant*(1+9*n**4)*n**p
        bound = n*((1+9*n**4)/beta)**s
        assert bound <= delta/8*(1+1e-12)
        for k in range(1, n):
            selected_sets = [z for z in subsets(n) if len(z) == k]
            covers = [z for z in selected_sets if all(u in z or v in z for u, v, _ in edges)]
            objectives = [trace_function(grounded(l, z, np.inf), lambda x: x**(-s))
                          for z in selected_sets]
            best = min(objectives)
            baseline = (n-k)/3**s
            assert bool(covers) == (abs(best-baseline) < 1e-10)
            # The singleton-seed Turing reduction checks the returned sets,
            # rather than comparing real-valued objective thresholds.
            seeded_answers = []
            for seed in range(n):
                eligible = [j for j, z in enumerate(selected_sets) if seed in z]
                j = min(eligible, key=lambda index: objectives[index])
                seeded_answers.append(selected_sets[j])
            assert bool(covers) == any(all(u in z or v in z for u, v, _ in edges)
                                       for z in seeded_answers)
            if covers:
                # For every cover, an interval containing its soft value lies
                # strictly below every interval for a noncover at this beta.
                for z, value in zip(selected_sets, objectives):
                    if z not in covers:
                        assert baseline+bound < value-1e-10
            else:
                assert best >= baseline+delta-1e-10
        # Fixed beta_0, rational weights beta_0/B: exact entrywise identity.
        beta0 = Fraction(3, 2)
        weight = beta0 / beta
        z = {0, 1, 2}
        for i in range(n):
            for j in range(n):
                left = weight*int(l[i, j]) + (beta0 if i == j and i in z else 0)
                right = weight*(int(l[i, j]) + (beta if i == j and i in z else 0))
                assert left == right
    # Numerically check the same scaling at well-conditioned finite B.
    for beta in [10, 1000]:
        scale = 1.5/beta
        a = grounded(l, {0, 1, 2}, beta)
        b = grounded(scale*l, {0, 1, 2}, 1.5)
        for s in [.25, .5, .75]:
            lhs = trace_function(b, lambda x: x**(-s))
            rhs = scale**(-s)*trace_function(a, lambda x: x**(-s))
            assert abs(lhs-rhs) <= 1e-9*max(lhs, rhs)
    print("PASS T2 hardness reduction: exhaustive k=1..5, s={.25,.5,.75}; "
          "hard gap tolerance=1e-10; singleton-seed reduction checked; polynomial-B error <= delta/8; "
          "fixed-beta matrix scaling exact rational, trace relative tolerance=1e-9")


run("T2 hardness reduction", test)

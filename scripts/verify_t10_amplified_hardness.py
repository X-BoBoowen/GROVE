"""Constants and budget/extraction checks for the amplified gap reduction.

The underlying path/eigenvalue inequalities, including internal anchors, are
checked in verify_t10_paths.py. No finite test purports to prove NP-hardness.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))
from scipy.optimize import brentq
from fractions import Fraction
import math


def test():
    limit = (2*np.arccos(1/3)/np.pi)**2
    assert limit < 1-1/np.e < 2/3
    f = lambda x: 1-6*x+5*x*x-x*x*x
    assert f(Fraction(99,500)) > 0 and f(Fraction(1,5)) < 0
    assert 3*f(Fraction(309,2500)) < 1
    assert Fraction(103,165) < Fraction(5,8)
    assert Fraction(65,24) > Fraction(8,3)
    regimes, extracted = 0, 0
    for gamma in [limit+.001, .62, 1-1/np.e, .65, .75, .99, 1.]:
        for ell in range(2, 513):
            ta = np.pi/(2*ell-1)
            tb = brentq(lambda t: 3*np.cos((ell-.5)*t)-np.cos(t/2), 0., ta, xtol=1e-15)
            aa, bb = 4*np.sin(ta/2)**2, 4*np.sin(tb/2)**2
            if bb/aa < gamma:
                break
        else:
            raise AssertionError('No finite path length found')
        gap = gamma*aa-bb
        c = math.ceil(64/gap)+9
        assert c > max(8, 32/gap)
        alpha, beta = aa/(1+4/(3*c)), bb+24/(c*(ell-1))
        assert gamma*alpha > beta
        assert beta/alpha > bb/aa > limit
        assert gamma*alpha-beta > gap/2
        for n in [4, 6, 20, 100]:
            m = 3*n//2
            for k in [1, n//2, n-1]:
                copies = c*k
                new_n = n+copies*m*(ell-1)
                new_m = copies*m*ell
                assert copies*(ell-1) >= 8*k
                assert aa/(1+2*(1+aa)/(3*copies)) >= alpha-1e-15
                assert bb+24*k/(copies*(ell-1)) <= beta+1e-15
                assert k < new_n and new_m <= c*ell*n*m
        for sigma in [Fraction(0), Fraction(1, 100), Fraction(2)]:
            w = math.floor((1-gamma)*(1+float(sigma))/(gamma*alpha-beta))+1
            yes = w*alpha+1+float(sigma)
            no = w*beta+1+float(sigma)
            assert gamma*yes > no
        regimes += 1
    # Extraction uses the original part of any output, not an assumed root-only
    # feasible family. A cover with fewer than k originals can be padded.
    for n, edges in [(4, [(i,j) for i in range(4) for j in range(i+1,4)]),
                     (6, [(i,j) for i in range(3) for j in range(3,6)])]:
        for k in range(1,n):
            yes = any(len(s)<=k and all(u in s or v in s for u,v in edges) for s in subsets(n))
            for roots in subsets(n):
                if len(roots)>k:
                    continue
                # Fill remaining slots with arbitrary distinct internal labels.
                output = set(roots) | set(range(n,n+k-len(roots)))
                assert len(output)==k and output & set(range(n)) == set(roots)
                covered = all(u in roots or v in roots for u,v in edges)
                if covered:
                    assert yes
                    padded = set(roots)
                    for u in range(n):
                        if len(padded)<k:
                            padded.add(u)
                    assert len(padded)==k and all(u in padded or v in padded for u,v in edges)
                else:
                    assert any(u not in roots and v not in roots for u,v in edges)
                extracted += 1
    print(f'PASS T10 amplified hardness: {regimes} separated target factors including 1-1/e; '
          f'{extracted} exact-budget extraction cases; seeded shifts {{0,.01,2}}; '
          f'c*={limit:.10f}; scalar tolerance=1e-15')


run('T10 amplified hardness', test)

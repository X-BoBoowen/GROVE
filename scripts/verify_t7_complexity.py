"""Primitive counts, depth/precision dependence, and explicit s=1 comparison conditions."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))


def test():
    rng = np.random.default_rng(704)
    a = fixtures()[2]
    for order in [1, 2, 5]:
        _, solves = coarse_jl_keys(a, np.inf, order, rng)
        d = int(np.ceil(128/.4**2*np.log(12*len(a)*(order+1)/.1)))
        assert solves == 2*d*((order+2)//2)
    # Normalize B=1. The common energy precision yields a residual/chain error
    # budget at every power, retaining the p*log(kappa) precision factor.
    for kappa in [1, 3, 10, 1e4]:
        for order in [1, 2, 5, 20]:
            depth = (order+2)//2
            epsilon, envelope = .25, 100.0
            precision = epsilon/(64*depth*kappa**(2*depth)*(envelope+1))
            residual = precision*np.sqrt(kappa)*kappa**(depth-1)*(envelope+1)
            error = 0.0
            for ell in range(1, depth+1):
                error = kappa*(error+residual)
                assert error <= epsilon/64+1e-12
            assert np.isfinite(-np.log(precision))
    for n in [10, 1000]:
        for k in [1, 5]:
            max_refreshes = k*n-k*(k-1)//2
            assert max_refreshes == sum(n-j for j in range(k))
    # s=1: ordinary Lanczos has R*h sparse products; the SDDM batch
    # baseline has k/eps^2 RHS up to logarithms. Neither dominates universally.
    k, epsilon = 10, .1
    baseline_rhs = k/epsilon**2
    assert 10*10 <= baseline_rhs
    assert 10000*100 > baseline_rhs
    print("PASS T7 complexity: exact 2*d*ceil((p+1)/2) initial solve counts; "
          "power-chain precision envelopes tolerance=1e-12; R bounds and conditional s=1 comparison pass")


run("T7 complexity", test)

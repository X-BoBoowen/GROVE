"""Schatten inverse-power convergence, refined eigenvalue-gap rate, and surrogate minimizers."""
from pathlib import Path
exec(compile(Path(__file__).with_name("_phase3_utils.py").read_text(), "_phase3_utils.py", "exec"))
from scipy.special import logsumexp


def test():
    worst, cases = 0.0, 0
    for spectrum in [np.array([.7]), np.array([.2, .2, .5, 1, 3]), np.geomspace(.1, 4, 8), np.ones(6)]:
        dimension, minimum = len(spectrum), spectrum[0]
        multiplicity = np.count_nonzero(spectrum == minimum)
        for s in [.5, 1, 2, 8, 32, 128]:
            log_ratio = logsumexp(-s*np.log(spectrum))/s+np.log(minimum)
            assert -2e-12 <= log_ratio <= np.log(dimension)/s+2e-12
            gap_ratio = 0 if multiplicity == dimension else minimum/spectrum[multiplicity]
            refined = np.log(multiplicity+(dimension-multiplicity)*gap_ratio**s)/s
            assert np.log(multiplicity)/s-2e-12 <= log_ratio <= refined+2e-12
            excess = log_ratio-np.log(multiplicity)/s
            assert excess <= (dimension-multiplicity)/multiplicity*gap_ratio**s/s+2e-12
            worst = max(worst, max(-log_ratio, log_ratio-refined, 0))
            cases += 1
        for eta in [.1, .5]:
            s = max(1, int(np.ceil(np.log(dimension)/np.log1p(eta))))
            ratio = np.exp(logsumexp(-s*np.log(spectrum))/s)*minimum
            assert ratio <= 1+eta+2e-12
    a = fixtures()[2]
    n, budget = len(a), 2
    eligible = [z for z in subsets(n) if len(z) == budget]
    lambdas = {z: np.linalg.eigvalsh(grounded(a, z, np.inf))[0] for z in eligible}
    best_lambda = max(lambdas.values())
    for s in [1, 2, 4, 16]:
        phi = {z: trace_function(grounded(a, z, np.inf), lambda x: x**(-s)) for z in eligible}
        minimum_phi = min(phi.values())
        for alpha in [1, 1.25, 2]:
            for z in eligible:
                if phi[z] <= alpha*minimum_phi*(1+1e-12):
                    assert lambdas[z] >= best_lambda/(alpha*(n-budget))**(1/s)-1e-10
    print(f"PASS T9 spectral limit: {cases} spectral/rate cases and exhaustive surrogate comparisons; "
          f"max log-bound violation={worst:.3e} < 2e-12; optimization tolerance=1e-10")


run("T9 spectral limit", test)

"""Finite hard grounding versus the weighted Zhou Eq. (4) score."""
from pathlib import Path
exec(compile(Path(__file__).with_name('_verify_utils.py').read_text(), '_verify_utils.py', 'exec'))


def test():
    cases = 0
    smallest_ratio = 1.0
    for t in [.2, .1, .01, .001, .0001, .00001, .000001]:
        # Vertex order z,a,b,c; the follower graph remains connected.
        full = laplacian(4, [(0, 1, t), (1, 2, t), (0, 2, 1),
                             (2, 3, 1), (3, 0, 1)])
        x = full[1:, 1:]
        vals, vecs = np.linalg.eigh(x)
        u = np.abs(vecs[:, 0])
        assert np.all(u > 0)
        adjacency = np.diag(np.diag(x)) - x
        score = 2 * u * (adjacency @ u)
        assert np.argmax(score) == 1
        assert abs(score[1] - score[0] - score[2]) < 1e-12
        spectra = [np.linalg.eigvalsh(updated(x, v, np.inf)) for v in range(3)]
        spectral_values = np.array([s[0] for s in spectra])
        assert np.argmax(spectral_values) == 0
        assert spectra[0][0] >= 1 - 1e-12
        assert abs(spectra[1][0] - 2*t) < 1e-12
        ratio = spectral_values[1] / spectral_values[0]
        assert ratio <= 2*t + 1e-12
        smallest_ratio = min(ratio, smallest_ratio)
        for p in [1, 1.5, 2, 4, 8, 16]:
            objectives = [np.sum(s**(-p)) for s in spectra]
            assert np.argmin(objectives) == 0
            assert objectives[0] <= 2 + 1e-10
            cases += 1
    print(f'PASS Zhou response: {cases} power/weight checks; unique score choices and spectral optima; '
          f'smallest score/OPT ratio={smallest_ratio:.6g}; tolerance=1e-10')


run('Zhou finite-removal response', test)

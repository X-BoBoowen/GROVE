# GROVE

**Reusing solution sketches for current-flow group selection.**

Given a connected graph and a budget `k`, GROVE greedily selects a group of `k`
grounded nodes that maximizes current-flow group closeness, i.e. minimizes
`tr(L_{-S}^{-1})`, the trace of the inverse grounded Laplacian. It keeps two
Johnson–Lindenstrauss solution sketches across greedy rounds, updates them by
the rank-one grounding identity, and refreshes only the candidates that can
still win a round.

## Layout

```
paper/     LaTeX sources; main.pdf (submission) and full_version.pdf (with proof appendices)
src/       ground.cpp      engine: GROVE (inc) and the ApproxGreedy (jlg), forest, Zhou baselines
           schur_cfcm.cpp  SchurCFCM baseline
           validate.py     checks the engine against dense linear algebra on a small graph
scripts/   numerical checks of every lemma and theorem (one script per claim), figure data
runs/      raw outputs of the experiments in the paper and the shell drivers that produced them
pilot/     early small-graph sensitivity experiments
notes/     theory.tex: working proofs behind the appendices
```

## Build and run

```bash
g++ -O3 -march=native -std=c++17 -fopenmp src/ground.cpp -o ground
g++ -O3 -march=native -std=c++17 -fopenmp src/schur_cfcm.cpp -o schur_cfcm

./ground convert com-dblp.ungraph.txt com-dblp.bin            # SNAP edge list -> binary graph
./ground seed com-dblp.bin --d 128                            # first node (JL estimate of argmin L^+_vv)
./ground inc  com-dblp.bin --p 1 --k 20 --seed <v> --d 64     # GROVE
./ground jlg  com-dblp.bin --p 1 --k 20 --seed <v> --d 64     # ApproxGreedy
./ground eval com-dblp.bin --set selected.txt --probes 64     # evaluate a selected set
python src/validate.py ./ground small.bin
```

The drivers in `runs/*.sh` list the exact command lines of every reported run;
they assume the binaries and a `bin/` directory of converted SNAP graphs in the
working directory.

Verification scripts need only NumPy and SciPy:

```bash
uv run -q --no-project --with numpy --with scipy python -I scripts/check_grounding_supermodular.py
```

`scripts/fig_error_trajectory.py` takes the path of SNAP's `ca-HepPh.txt` as its argument.

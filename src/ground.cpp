// Grounded group selection engine (paper 9, phase 2).
// Hard grounding of the graph Laplacian: X_S = L restricted to V \ S (sigma = 0, as in CFCC).
// Objective Phi_p(S) = tr X_S^{-p} for integer p >= 1; p = 1 is group CFCC (C(S) = n / Phi_1).
//
// Modes:
//   convert <edges.txt[.gz]> <out.bin>          undirected simple LCC in binary CSR
//   seed    <g.bin> [--d D]                       argmin_v L^+_vv by JL (common first anchor)
//   celf    <g.bin> --p P --k K --seed V          JL upper bounds at S0={V}, lazy exact refresh
//   jlg     <g.bin> --p P --k K --seed V          fresh all-node JL every round (p=1: ApproxGreedy)
//   eval    <g.bin> --set file [--probes N]       tr X^-1, tr X^-2 (Hutchinson, common probes), lambda_min
//   dense   <g.bin> --p P --k K --seed V          exact dense greedy (small n only, validation)
//   exactmarg <g.bin> --p P --seed V              print exact refresh marginals for all nodes (validation)
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
typedef long long ll;

static double now_s() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }

static inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
static inline double rsign(uint64_t a, uint64_t b, uint64_t c) {
  return (mix64(a ^ mix64(b ^ mix64(c))) & 1ULL) ? 1.0 : -1.0;
}

struct Rng { uint64_t s; uint64_t next() { s += 0x9e3779b97f4a7c15ULL; return mix64(s); } };

struct Graph {
  int n = 0;
  vector<ll> ptr;
  vector<int> adj;
  vector<double> deg;
  ll nnz() const { return (ll)adj.size(); }
};

// ---------------------------------------------------------------- conversion
static void convert(const string& in, const string& out) {
  double t0 = now_s();
  FILE* f;
  bool gz = in.size() > 3 && in.substr(in.size() - 3) == ".gz";
  if (gz) f = popen(("zcat '" + in + "'").c_str(), "r");
  else f = fopen(in.c_str(), "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", in.c_str()); exit(1); }
  vector<ll> ea, eb;
  static char buf[1 << 16];
  while (fgets(buf, sizeof buf, f)) {
    char* s = buf;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '#' || *s == '%' || *s == '\n' || *s == 0) continue;
    char* e;
    ll a = strtoll(s, &e, 10);
    if (e == s) continue;
    s = e;
    ll b = strtoll(s, &e, 10);
    if (e == s) continue;
    if (a != b) { ea.push_back(a); eb.push_back(b); }
  }
  if (gz) pclose(f); else fclose(f);
  vector<ll> ids(ea);
  ids.insert(ids.end(), eb.begin(), eb.end());
  sort(ids.begin(), ids.end());
  ids.erase(unique(ids.begin(), ids.end()), ids.end());
  int N = (int)ids.size();
  size_t m = ea.size();
  vector<int> u(m), v(m);
#pragma omp parallel for
  for (size_t i = 0; i < m; i++) {
    u[i] = (int)(lower_bound(ids.begin(), ids.end(), ea[i]) - ids.begin());
    v[i] = (int)(lower_bound(ids.begin(), ids.end(), eb[i]) - ids.begin());
  }
  vector<ll>().swap(ea); vector<ll>().swap(eb);
  vector<ll> cnt(N + 1, 0);
  for (size_t i = 0; i < m; i++) { cnt[u[i] + 1]++; cnt[v[i] + 1]++; }
  for (int i = 0; i < N; i++) cnt[i + 1] += cnt[i];
  vector<int> a2(cnt[N]);
  vector<ll> pos(cnt.begin(), cnt.end() - 1);
  for (size_t i = 0; i < m; i++) { a2[pos[u[i]]++] = v[i]; a2[pos[v[i]]++] = u[i]; }
  vector<int> dg(N);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    sort(a2.begin() + cnt[i], a2.begin() + cnt[i + 1]);
    dg[i] = (int)(unique(a2.begin() + cnt[i], a2.begin() + cnt[i + 1]) - (a2.begin() + cnt[i]));
  }
  // largest connected component by BFS
  vector<int> comp(N, -1);
  int best = -1, bestsz = 0, nc = 0;
  vector<int> q;
  for (int s = 0; s < N; s++) {
    if (comp[s] >= 0) continue;
    q.clear(); q.push_back(s); comp[s] = nc;
    for (size_t h = 0; h < q.size(); h++) {
      int x = q[h];
      for (ll a = cnt[x]; a < cnt[x] + dg[x]; a++) if (comp[a2[a]] < 0) { comp[a2[a]] = nc; q.push_back(a2[a]); }
    }
    if ((int)q.size() > bestsz) { bestsz = (int)q.size(); best = nc; }
    nc++;
  }
  // Relabel the LCC in Cuthill--McKee order (BFS from a minimum-degree vertex, neighbours by increasing
  // degree) for memory locality of the block SpMV.
  vector<int> lab(N, -1);
  int n = 0;
  {
    int start = -1;
    for (int i = 0; i < N; i++) if (comp[i] == best && (start < 0 || dg[i] < dg[start])) start = i;
    vector<int> order{start};
    lab[start] = n++;
    vector<int> nb;
    for (size_t h = 0; h < order.size(); h++) {
      int x = order[h];
      nb.clear();
      for (ll a = cnt[x]; a < cnt[x] + dg[x]; a++) if (lab[a2[a]] < 0) nb.push_back(a2[a]);
      sort(nb.begin(), nb.end(), [&](int p, int q) { return dg[p] < dg[q] || (dg[p] == dg[q] && p < q); });
      for (int y : nb) if (lab[y] < 0) { lab[y] = n++; order.push_back(y); }
    }
  }
  vector<ll> ptr(n + 1, 0);
  for (int i = 0; i < N; i++) if (lab[i] >= 0) ptr[lab[i] + 1] = dg[i];
  for (int i = 0; i < n; i++) ptr[i + 1] += ptr[i];
  vector<int> adj(ptr[n]);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int i = 0; i < N; i++) {
    if (lab[i] < 0) continue;
    ll o = ptr[lab[i]];
    for (ll a = cnt[i]; a < cnt[i] + dg[i]; a++) adj[o++] = lab[a2[a]];
    sort(adj.begin() + ptr[lab[i]], adj.begin() + o);
  }
  FILE* g = fopen(out.c_str(), "wb");
  ll nnz = ptr[n];
  fwrite("GRND", 1, 4, g);
  fwrite(&n, sizeof n, 1, g);
  fwrite(&nnz, sizeof nnz, 1, g);
  fwrite(ptr.data(), sizeof(ll), n + 1, g);
  fwrite(adj.data(), sizeof(int), nnz, g);
  fclose(g);
  printf("{\"mode\":\"convert\",\"raw_nodes\":%d,\"raw_lines\":%zu,\"n\":%d,\"m\":%lld,\"components\":%d,\"time\":%.2f}\n",
         N, m, n, nnz / 2, nc, now_s() - t0);
}

static Graph load(const string& path) {
  Graph G;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  char mg[4];
  ll nnz;
  if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "GRND", 4)) { fprintf(stderr, "bad file\n"); exit(1); }
  if (fread(&G.n, sizeof G.n, 1, f) != 1 || fread(&nnz, sizeof nnz, 1, f) != 1) exit(1);
  G.ptr.resize(G.n + 1);
  G.adj.resize(nnz);
  if (fread(G.ptr.data(), sizeof(ll), G.n + 1, f) != (size_t)G.n + 1) exit(1);
  if (fread(G.adj.data(), sizeof(int), nnz, f) != (size_t)nnz) exit(1);
  fclose(f);
  G.deg.resize(G.n);
  for (int i = 0; i < G.n; i++) G.deg[i] = (double)(G.ptr[i + 1] - G.ptr[i]);
  return G;
}


// ---------------------------------------------------------------- approximate Cholesky preconditioner
// Kyng--Sachdeva approximate Gaussian elimination in the form of Laplacians.jl approxchol (random clique
// sampling, f-recursion, lazy minimum-degree order). Built once for the grounded Laplacian of the seed set S0:
// S0 is contracted to one ground vertex g = n that is never eliminated. The preconditioner for X_S, S >= S0,
// is r -> mask_S (Q Ltilde^+ Q^T r), which is symmetric positive definite; X_S differs from X_S0 (padded by the
// identity on S \ S0) by a perturbation of rank <= 2|S \ S0|.
struct ApproxChol {
  int n = 0;
  vector<int> col, rowv;
  vector<ll> cptr;
  vector<double> fval, d;
  double build_time = 0;
  ll fill = 0;
  void build(const Graph& G, const vector<char>& gr, uint64_t seed) {
    double t0 = now_s();
    n = G.n;
    int g = n;
    vector<vector<pair<int, double>>> E(n + 1);
    for (int u = 0; u < n; u++) {
      if (gr[u]) continue;
      double toS = 0;
      for (ll e = G.ptr[u]; e < G.ptr[u + 1]; e++) {
        int w = G.adj[e];
        if (gr[w]) toS += 1; else E[u].push_back({w, 1.0});
      }
      if (toS > 0) E[u].push_back({g, toS});
    }
    vector<char> elim(n + 1, 0);
    d.assign(n + 1, 0.0);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<pair<ll, int>>> pq;
    for (int u = 0; u < n; u++) if (!gr[u]) pq.push({(ll)E[u].size(), u});
    Rng rng{seed};
    vector<pair<int, double>> tmp;
    vector<double> cs;
    cptr.push_back(0);
    while (!pq.empty()) {
      auto [sz, v] = pq.top(); pq.pop();
      if (elim[v]) continue;
      if (sz != (ll)E[v].size()) { pq.push({(ll)E[v].size(), v}); continue; }
      tmp.clear();
      for (auto& e : E[v]) if (!elim[e.first]) tmp.push_back(e);
      sort(tmp.begin(), tmp.end());
      size_t m2 = 0;
      for (size_t i = 0; i < tmp.size(); i++) {
        if (m2 > 0 && tmp[m2 - 1].first == tmp[i].first) tmp[m2 - 1].second += tmp[i].second;
        else tmp[m2++] = tmp[i];
      }
      tmp.resize(m2);
      elim[v] = 1;
      vector<pair<int, double>>().swap(E[v]);
      col.push_back(v);
      int len = (int)tmp.size();
      if (len == 0) { cptr.push_back((ll)rowv.size()); d[v] = 0; continue; }
      sort(tmp.begin(), tmp.end(), [](const pair<int, double>& a, const pair<int, double>& b) { return a.second < b.second; });
      cs.resize(len);
      double acc = 0;
      for (int i = 0; i < len; i++) { acc += tmp[i].second; cs[i] = acc; }
      double wdeg = cs[len - 1], colScale = 1;
      for (int jo = 0; jo < len - 1; jo++) {
        int j = tmp[jo].first;
        double w = tmp[jo].second * colScale, f = w / wdeg;
        double r = (double)(rng.next() >> 11) * (1.0 / 9007199254740992.0) * (cs[len - 1] - cs[jo]) + cs[jo];
        int ko = (int)(lower_bound(cs.begin() + jo + 1, cs.end(), r) - cs.begin());
        if (ko >= len) ko = len - 1;
        int k = tmp[ko].first;
        double nv = f * (1 - f) * wdeg;
        if (j != g) E[j].push_back({k, nv});
        if (k != g) E[k].push_back({j, nv});
        colScale *= (1 - f);
        wdeg *= (1 - f) * (1 - f);
        rowv.push_back(j); fval.push_back(f);
      }
      rowv.push_back(tmp[len - 1].first); fval.push_back(1.0);
      d[v] = tmp[len - 1].second * colScale;
      cptr.push_back((ll)rowv.size());
    }
    fill = (ll)rowv.size();
    build_time = now_s() - t0;
  }
  // W interleaved columns: entry (i, c) at i * W + c; one sweep over the factor for all W columns
  void apply_block(int W, double* y) const {
    if (W == 1) { apply(y); return; }
    int nc = (int)col.size();
    vector<double> yi(W);
    for (int ii = 0; ii < nc; ii++) {
      int i = col[ii];
      ll j0 = cptr[ii], j1 = cptr[ii + 1];
      if (j0 == j1) continue;
      double* Yi = y + (size_t)i * W;
      for (int c = 0; c < W; c++) yi[c] = Yi[c];
      for (ll jj = j0; jj < j1 - 1; jj++) {
        double f = fval[jj], g1 = 1 - f; double* Yr = y + (size_t)rowv[jj] * W;
        for (int c = 0; c < W; c++) { Yr[c] += f * yi[c]; yi[c] *= g1; }
      }
      double* Yl = y + (size_t)rowv[j1 - 1] * W;
      for (int c = 0; c < W; c++) { Yl[c] += yi[c]; Yi[c] = yi[c]; }
    }
    for (int i = 0; i <= n; i++) { double di = d[i]; double* Yi = y + (size_t)i * W;
      for (int c = 0; c < W; c++) Yi[c] = di != 0 ? Yi[c] / di : 0.0; }
    for (int ii = nc - 1; ii >= 0; ii--) {
      int i = col[ii];
      ll j0 = cptr[ii], j1 = cptr[ii + 1];
      if (j0 == j1) continue;
      double* Yl = y + (size_t)rowv[j1 - 1] * W; double* Yi = y + (size_t)i * W;
      for (int c = 0; c < W; c++) yi[c] = Yi[c] + Yl[c];
      for (ll jj = j1 - 2; jj >= j0; jj--) {
        double f = fval[jj], g1 = 1 - f; const double* Yr = y + (size_t)rowv[jj] * W;
        for (int c = 0; c < W; c++) yi[c] = g1 * yi[c] + f * Yr[c];
      }
      for (int c = 0; c < W; c++) Yi[c] = yi[c];
    }
  }
  // y has n + 1 entries (ground last); on return y holds the grounded solution on V (y[g] = 0)
  void apply(double* y) const {
    int nc = (int)col.size();
    for (int ii = 0; ii < nc; ii++) {
      int i = col[ii];
      ll j0 = cptr[ii], j1 = cptr[ii + 1];
      if (j0 == j1) continue;
      double yi = y[i];
      for (ll jj = j0; jj < j1 - 1; jj++) { y[rowv[jj]] += fval[jj] * yi; yi *= (1 - fval[jj]); }
      y[rowv[j1 - 1]] += yi;
      y[i] = yi;
    }
    for (int i = 0; i <= n; i++) y[i] = d[i] != 0 ? y[i] / d[i] : 0.0;
    for (int ii = nc - 1; ii >= 0; ii--) {
      int i = col[ii];
      ll j0 = cptr[ii], j1 = cptr[ii + 1];
      if (j0 == j1) continue;
      double yi = y[i] + y[rowv[j1 - 1]];
      for (ll jj = j1 - 2; jj >= j0; jj--) yi = (1 - fval[jj]) * yi + fval[jj] * y[rowv[jj]];
      y[i] = yi;
    }
  }
};

// ---------------------------------------------------------------- block PCG
// Layout: block of B columns, entry (i, c) at i * B + c. Grounded rows are kept exactly zero.
struct Solver {
  const Graph& G;
  vector<char> gr;      // grounded mask
  vector<double> dinv;  // Jacobi preconditioner on alive rows, 0 on grounded rows
  bool full = false;    // full Laplacian (no grounding): RHS and solution projected onto 1-perp
  ll spmv_cols = 0;     // number of single-column operator applications
  int threads;
  const ApproxChol* ac = nullptr;  // optional preconditioner (replaces Jacobi)
  // z = M^{-1} r for B interleaved columns; rz[c] = <r_c, z_c>
  int acmin = 4;
  void precond(int B, const double* r, double* z, double* rz) {
    int n = G.n;
    if (!ac || B < acmin) {
      vector<double> tot(B, 0.0);
#pragma omp parallel
      {
        vector<double> loc(B, 0.0);
#pragma omp for nowait
        for (int i = 0; i < n; i++)
          for (int c = 0; c < B; c++) { size_t a = (size_t)i * B + c; z[a] = dinv[i] * r[a]; loc[c] += r[a] * z[a]; }
#pragma omp critical
        for (int c = 0; c < B; c++) tot[c] += loc[c];
      }
      for (int c = 0; c < B; c++) rz[c] = tot[c];
      return;
    }
    int T = max(1, min(threads, B)), W = (B + T - 1) / T, ng = (B + W - 1) / W;
#pragma omp parallel num_threads(ng)
    {
#pragma omp for schedule(static, 1)
      for (int gidx = 0; gidx < ng; gidx++) {
        int c0 = gidx * W, w = min(W, B - c0);
        vector<double> y((size_t)(n + 1) * w);
        for (int i = 0; i < n; i++) for (int c = 0; c < w; c++) y[(size_t)i * w + c] = gr[i] ? 0.0 : r[(size_t)i * B + c0 + c];
        for (int c = 0; c < w; c++) y[(size_t)n * w + c] = 0;
        ac->apply_block(w, y.data());
        for (int c = 0; c < w; c++) {
          double t = 0;
          for (int i = 0; i < n; i++) { double v = gr[i] ? 0.0 : y[(size_t)i * w + c]; z[(size_t)i * B + c0 + c] = v; t += v * r[(size_t)i * B + c0 + c]; }
          rz[c0 + c] = t;
        }
      }
    }
  }
  Solver(const Graph& g) : G(g), gr(g.n, 0), dinv(g.n), threads(omp_get_max_threads()) {
    for (int i = 0; i < G.n; i++) dinv[i] = 1.0 / G.deg[i];
  }
  void ground(int v) { gr[v] = 1; dinv[v] = 0; }

  // y = X x (block), and acc[c] = sum_i x_ic y_ic
  void apply(int B, const double* x, double* y, double* acc) {
    int n = G.n;
    vector<double> tot(B, 0.0);
#pragma omp parallel
    {
      vector<double> loc(B, 0.0);
#pragma omp for schedule(dynamic, 2048) nowait
      for (int i = 0; i < n; i++) {
        double* yi = y + (size_t)i * B;
        if (gr[i]) { for (int c = 0; c < B; c++) yi[c] = 0; continue; }
        const double* xi = x + (size_t)i * B;
        double d = G.deg[i];
        for (int c = 0; c < B; c++) yi[c] = d * xi[c];
        for (ll a = G.ptr[i]; a < G.ptr[i + 1]; a++) {
          const double* xj = x + (size_t)G.adj[a] * B;
#pragma omp simd
          for (int c = 0; c < B; c++) yi[c] -= xj[c];
        }
        if (acc) for (int c = 0; c < B; c++) loc[c] += xi[c] * yi[c];
      }
      if (acc) {
#pragma omp critical
        for (int c = 0; c < B; c++) tot[c] += loc[c];
      }
    }
    if (acc) for (int c = 0; c < B; c++) acc[c] = tot[c];
    spmv_cols += B;
  }

  void project(int B, double* x) {  // remove the mean of every column (full Laplacian only)
    int n = G.n;
    vector<double> s(B, 0.0);
    for (int i = 0; i < n; i++) for (int c = 0; c < B; c++) s[c] += x[(size_t)i * B + c];
    for (int c = 0; c < B; c++) s[c] /= n;
#pragma omp parallel for
    for (int i = 0; i < n; i++) for (int c = 0; c < B; c++) x[(size_t)i * B + c] -= s[c];
  }

  // Solve X x = b column-wise to relative residual tol; x is overwritten (zero start). Returns iterations.
  int solve(int B, const double* b, double* x, double tol, int maxit = 20000) {
    int n = G.n;
    size_t N = (size_t)n * B;
    vector<double> r(b, b + N), z(N), p(N), q(N);
    fill(x, x + N, 0.0);
    if (full) project(B, r.data());
    vector<double> bn(B, 0), rz(B, 0), rr(B, 0), pq(B), alpha(B), beta(B);
    vector<char> done(B, 0);
    precond(B, r.data(), z.data(), rz.data());
#pragma omp parallel
    {
      vector<double> l1(B, 0);
#pragma omp for nowait
      for (int i = 0; i < n; i++)
        for (int c = 0; c < B; c++) {
          size_t a = (size_t)i * B + c;
          p[a] = z[a];
          l1[c] += r[a] * r[a];
        }
#pragma omp critical
      for (int c = 0; c < B; c++) bn[c] += l1[c];
    }
    for (int c = 0; c < B; c++) { bn[c] = sqrt(bn[c]); if (bn[c] == 0) done[c] = 1; }
    int it = 0;
    for (; it < maxit; it++) {
      bool all = true;
      for (int c = 0; c < B; c++) all = all && done[c];
      if (all) break;
      apply(B, p.data(), q.data(), pq.data());
      for (int c = 0; c < B; c++) alpha[c] = (done[c] || pq[c] <= 0) ? 0.0 : rz[c] / pq[c];
      vector<double> nrr(B, 0), nrz(B, 0);
#pragma omp parallel
      {
        vector<double> l1(B, 0);
#pragma omp for nowait
        for (int i = 0; i < n; i++) {
          for (int c = 0; c < B; c++) {
            size_t a = (size_t)i * B + c;
            x[a] += alpha[c] * p[a];
            r[a] -= alpha[c] * q[a];
            l1[c] += r[a] * r[a];
          }
        }
#pragma omp critical
        for (int c = 0; c < B; c++) nrr[c] += l1[c];
      }
      precond(B, r.data(), z.data(), nrz.data());
      for (int c = 0; c < B; c++) {
        if (done[c]) { beta[c] = 0; continue; }
        if (sqrt(nrr[c]) <= tol * bn[c]) { done[c] = 1; beta[c] = 0; continue; }
        beta[c] = nrz[c] / rz[c];
        rz[c] = nrz[c];
      }
#pragma omp parallel for
      for (int i = 0; i < n; i++)
        for (int c = 0; c < B; c++) {
          size_t a = (size_t)i * B + c;
          if (!done[c]) p[a] = z[a] + beta[c] * p[a];
        }
    }
    if (full) project(B, x);
    return it;
  }
};


static ApproxChol g_ac;
static int g_acmin = 4;
static void setup_ac(Solver& S, int use, uint64_t seed) {
  if (!use) return;
  g_ac.build(S.G, S.gr, seed);
  S.ac = &g_ac;
  S.acmin = g_acmin;
  fprintf(stderr, "approxchol: build %.2f s, factor entries %lld (m=%lld)\n", g_ac.build_time, g_ac.fill, S.G.nnz() / 2);
}

// ---------------------------------------------------------------- integer marginal from moments
// Hard grounding: f_j = (-1)^j r_{j+1}, j = 0..p; log-series recurrence; q_p = (-1)^p p g_p.
static double marginal_from_moments(const double* r /* r[1..p+1] */, int p) {
  vector<double> f(p + 1), g(p + 1, 0.0);
  for (int j = 0; j <= p; j++) f[j] = ((j & 1) ? -1.0 : 1.0) * r[j + 1];
  for (int k = 1; k <= p; k++) {
    double acc = k * f[k];
    for (int j = 1; j < k; j++) acc -= j * g[j] * f[k - j];
    g[k] = acc / (k * f[0]);
  }
  return ((p & 1) ? -1.0 : 1.0) * p * g[p];
}

// ---------------------------------------------------------------- JL estimation of moments r_1..r_{p+1}
// Odd moments r_{2i+1}(v) = || C_S X^{-(i+1)} e_v ||^2 via RHS P = C_S^T z (z Rademacher on edges).
// Even moments r_{2i}(v) = || X^{-i} e_v ||^2 via RHS Q (Rademacher on alive vertices).
// Full-Laplacian mode (seed): only r_1 = L^+_vv with the odd family.
struct JLStats { double time = 0; ll solves = 0; ll iters = 0; };

static vector<vector<double>> jl_moments(Solver& S, int pmax_moment, int d, int BS, double tol, uint64_t seed,
                                         JLStats& st, vector<double>* Ysk = nullptr) {
  const Graph& G = S.G;
  int n = G.n;
  double t0 = now_s();
  vector<vector<double>> R(pmax_moment + 1, vector<double>(n, 0.0));
  int odd_len = (pmax_moment - 1) / 2 + 1;  // chain length: r_1, r_3, ... <= pmax_moment
  int even_len = pmax_moment / 2;           // r_2, r_4, ...
  for (int c0 = 0; c0 < d; c0 += BS) {
    int B = min(BS, d - c0);
    size_t N = (size_t)n * B;
    vector<double> b(N), x(N);
    for (int fam = 0; fam < 2; fam++) {
      int len = fam == 0 ? odd_len : even_len;
      if (len == 0) continue;
#pragma omp parallel for schedule(dynamic, 2048)
      for (int i = 0; i < n; i++) {
        double* bi = b.data() + (size_t)i * B;
        if (S.gr[i]) { for (int c = 0; c < B; c++) bi[c] = 0; continue; }
        if (fam == 0) {
          for (int c = 0; c < B; c++) bi[c] = 0;
          for (ll a = G.ptr[i]; a < G.ptr[i + 1]; a++) {
            int j = G.adj[a];
            uint64_t lo = min(i, j), hi = max(i, j);
            double o = i < j ? 1.0 : -1.0;
            for (int c = 0; c < B; c++) bi[c] += o * rsign(seed, (uint64_t)(c0 + c), lo * 0x1000003ULL + hi);
          }
        } else {
          for (int c = 0; c < B; c++) bi[c] = rsign(seed ^ 0xabcdefULL, (uint64_t)(c0 + c), (uint64_t)i);
        }
      }
      for (int step = 1; step <= len; step++) {
        st.iters += S.solve(B, b.data(), x.data(), tol);
        st.solves += B;
        if (Ysk && step == 1) {  // keep Y = X^{-1} P (fam 0) and X^{-1} Q (fam 1)
          vector<double>& Y = Ysk[fam];
          if (Y.empty()) Y.assign((size_t)n * d, 0.0);
#pragma omp parallel for
          for (int i = 0; i < n; i++)
            for (int c = 0; c < B; c++) Y[(size_t)i * d + c0 + c] = x[(size_t)i * B + c];
        }
        int mom = fam == 0 ? 2 * step - 1 : 2 * step;
        if (mom <= pmax_moment) {
          vector<double>& Rm = R[mom];
#pragma omp parallel for
          for (int i = 0; i < n; i++) {
            double s = 0;
            const double* xi = x.data() + (size_t)i * B;
            for (int c = 0; c < B; c++) s += xi[c] * xi[c];
            Rm[i] += s;
          }
        }
        swap(b, x);
      }
    }
  }
  for (int j = 1; j <= pmax_moment; j++)
    for (int i = 0; i < n; i++) R[j][i] /= d;
  st.time += now_s() - t0;
  return R;
}

// ---------------------------------------------------------------- exact refresh of candidates
// For candidates v_1..v_B: y_1 = X^{-1} e_v, y_{i+1} = X^{-1} y_i. r_1 = y_1[v], r_{2i} = |y_i|^2,
// r_{2i+1} = <y_i, y_{i+1}>.
static vector<double> refresh(Solver& S, const vector<int>& cand, int p, double tol, ll& iters,
                              vector<vector<double>>* moms = nullptr, vector<vector<double>>* chain = nullptr) {
  int n = S.G.n, B = (int)cand.size();
  int len = (p + 2) / 2;  // need r_{p+1}
  size_t N = (size_t)n * B;
  vector<double> b(N, 0.0), x(N), prev;
  for (int c = 0; c < B; c++) b[(size_t)cand[c] * B + c] = 1.0;
  vector<vector<double>> r(B, vector<double>(2 * len + 2, 0.0));
  vector<vector<double>> steps_x;
  for (int step = 1; step <= len; step++) {
    iters += S.solve(B, b.data(), x.data(), tol);
    vector<double> sq(B, 0), dt(B, 0);
#pragma omp parallel
    {
      vector<double> l1(B, 0), l2(B, 0);
#pragma omp for nowait
      for (int i = 0; i < n; i++)
        for (int c = 0; c < B; c++) {
          size_t a = (size_t)i * B + c;
          l1[c] += x[a] * x[a];
          l2[c] += b[a] * x[a];  // <y_{step-1}, y_step>, y_0 = e_v
        }
#pragma omp critical
      for (int c = 0; c < B; c++) { sq[c] += l1[c]; dt[c] += l2[c]; }
    }
    for (int c = 0; c < B; c++) { r[c][2 * step - 1] = dt[c]; r[c][2 * step] = sq[c]; }
    if (chain) steps_x.push_back(x);
    swap(b, x);
  }
  vector<double> out(B);
  for (int c = 0; c < B; c++) out[c] = marginal_from_moments(r[c].data(), p);
  if (chain) {  // chain y_1..y_len of the batch maximizer (column-extracted)
    int cb = (int)(max_element(out.begin(), out.end()) - out.begin());
    chain->assign(len, vector<double>(n));
    for (int t = 0; t < len; t++) for (int i = 0; i < n; i++) (*chain)[t][i] = steps_x[t][(size_t)i * B + cb];
  }
  if (moms) *moms = r;
  return out;
}

// ---------------------------------------------------------------- arguments
struct Args {
  string mode, graph, setfile, dumpfile;
  int p = 1, k = 20, d = 64, BS = 32, RB = 16, seedv = -1, probes = 128, lanczos = 120, threads = 0;
  int budget = 0, sketch = 0, keepexact = 1;
  double tol = 1e-3, rtol = 1e-6, slack = 0.0, accept = 0.0, rebuild = 1.0, eps = 0.3, itol = 1e-9, zeta = 0.3;
  int forests = 0, maxforests = 1 << 20, checkupd = 0, exactinit = 0, keyub = 0, ac = 0, reuse = 1;
  uint64_t rng = 12345;
};
static Args parse(int argc, char** argv) {
  Args a;
  a.mode = argv[1];
  a.graph = argv[2];
  for (int i = 3; i < argc; i++) {
    string s = argv[i];
    auto nx = [&]() { return string(argv[++i]); };
    if (s == "--p") a.p = stoi(nx());
    else if (s == "--k") a.k = stoi(nx());
    else if (s == "--d") a.d = stoi(nx());
    else if (s == "--bs") a.BS = stoi(nx());
    else if (s == "--rb") a.RB = stoi(nx());
    else if (s == "--budget") a.budget = stoi(nx());
    else if (s == "--sketch") a.sketch = stoi(nx());
    else if (s == "--keepexact") a.keepexact = stoi(nx());
    else if (s == "--seed") a.seedv = stoi(nx());
    else if (s == "--probes") a.probes = stoi(nx());
    else if (s == "--lanczos") a.lanczos = stoi(nx());
    else if (s == "--tol") a.tol = stod(nx());
    else if (s == "--rtol") a.rtol = stod(nx());
    else if (s == "--slack") a.slack = stod(nx());
    else if (s == "--accept") a.accept = stod(nx());
    else if (s == "--rebuild") a.rebuild = stod(nx());
    else if (s == "--eps") a.eps = stod(nx());
    else if (s == "--itol") a.itol = stod(nx());
    else if (s == "--checkupd") a.checkupd = stoi(nx());
    else if (s == "--exactinit") a.exactinit = stoi(nx());
    else if (s == "--keyub") a.keyub = stoi(nx());
    else if (s == "--ac") a.ac = stoi(nx());
    else if (s == "--reuse") a.reuse = stoi(nx());
    else if (s == "--acmin") g_acmin = stoi(nx());
    else if (s == "--zeta") a.zeta = stod(nx());
    else if (s == "--forests") a.forests = stoi(nx());
    else if (s == "--maxforests") a.maxforests = stoi(nx());
    else if (s == "--rng") a.rng = stoull(nx());
    else if (s == "--set") a.setfile = nx();
    else if (s == "--dump") a.dumpfile = nx();
    else if (s == "--threads") a.threads = stoi(nx());
    else { fprintf(stderr, "unknown arg %s\n", s.c_str()); exit(1); }
  }
  return a;
}

static void print_set(const vector<int>& S) {
  printf("[");
  for (size_t i = 0; i < S.size(); i++) printf("%s%d", i ? "," : "", S[i]);
  printf("]");
}

// ---------------------------------------------------------------- modes
static int mode_seed(const Graph& G, const Args& a) {
  Solver S(G);
  S.full = true;
  JLStats st;
  auto R = jl_moments(S, 1, a.d, a.BS, a.tol, a.rng ^ 0x5eedULL, st);
  int best = 0;
  for (int i = 1; i < G.n; i++) if (R[1][i] < R[1][best]) best = i;
  printf("{\"mode\":\"seed\",\"n\":%d,\"m\":%lld,\"seed\":%d,\"Lpinv_vv\":%.6g,\"d\":%d,\"time\":%.3f,\"pcg_iters\":%lld}\n",
         G.n, G.nnz() / 2, best, R[1][best], a.d, st.time, st.iters);
  return best;
}

static void mode_celf(const Graph& G, const Args& a) {
  double t0 = now_s();
  Solver S(G);
  S.ground(a.seedv);
  setup_ac(S, a.ac, a.rng ^ 0xac0ULL);
  vector<int> sel{a.seedv};
  JLStats st;
  auto R = jl_moments(S, a.p + 1, a.d, a.BS, a.tol, a.rng, st);
  double t_init = now_s() - t0;
  fprintf(stderr, "init: %.2f s, pcg iters %lld\n", t_init, st.iters);
  struct It { double key; int v; int round; bool operator<(const It& o) const { return key < o.key; } };
  priority_queue<It> H;
  {
    vector<double> rv(a.p + 2);
    for (int i = 0; i < G.n; i++) {
      if (S.gr[i]) continue;
      for (int j = 1; j <= a.p + 1; j++) rv[j] = R[j][i];
      double q = marginal_from_moments(rv.data(), a.p);
      H.push({q * (1 + a.slack), i, -1});
    }
  }
  vector<vector<double>>().swap(R);
  ll refreshed = 0, riters = 0, batches = 0, rebuilds = 0;
  vector<double> round_time;
  vector<int> round_refresh;
  vector<double> gains;
  for (int round = 1; round < a.k; round++) {
    double tr = now_s();
    int rr = 0;
    // fresh = candidates refreshed at the current set; all heap keys are stale upper bounds.
    vector<It> fresh;
    int bf = -1;
    bool rebuilt = false;
    while (true) {
      double K = H.empty() ? -1.0 : H.top().key;
      if (bf >= 0 && fresh[bf].key >= (1.0 - a.accept) * K) break;
      if (!rebuilt && a.rebuild > 0 && (double)rr * ((a.p + 2) / 2) >= a.rebuild * (a.p + 1) * a.d) {
        // Lazy bounds are too loose this round: re-estimate every stale key with a fresh sketch at the
        // current set and keep the smaller of the stale exact bound and the inflated estimate.
        rebuilt = true; rebuilds++;
        auto R2 = jl_moments(S, a.p + 1, a.d, a.BS, a.tol, a.rng + 104729ULL * round, st);
        vector<It> items;
        while (!H.empty()) { items.push_back(H.top()); H.pop(); }
        vector<double> rv(a.p + 2);
        for (auto& it : items) {
          for (int j = 1; j <= a.p + 1; j++) rv[j] = R2[j][it.v];
          it.key = min(it.key, marginal_from_moments(rv.data(), a.p) * (1 + a.slack));
          H.push(it);
        }
        continue;
      }
      vector<int> cand;
      while (!H.empty() && (int)cand.size() < a.RB) { cand.push_back(H.top().v); H.pop(); }
      auto val = refresh(S, cand, a.p, a.rtol, riters);
      batches++;
      for (size_t c = 0; c < cand.size(); c++) {
        fresh.push_back({val[c], cand[c], round});
        if (bf < 0 || val[c] > fresh[bf].key) bf = (int)fresh.size() - 1;
      }
      refreshed += cand.size(); rr += cand.size();
    }
    S.ground(fresh[bf].v); sel.push_back(fresh[bf].v); gains.push_back(fresh[bf].key);
    for (int i = 0; i < (int)fresh.size(); i++) if (i != bf) H.push(fresh[i]);
    round_time.push_back(now_s() - tr);
    round_refresh.push_back(rr);
    fprintf(stderr, "round %d: v=%d gain=%.6g refreshed=%d time=%.2f total=%.2f refresh_iters=%lld\n", round,
            sel.back(), gains.back(), rr, round_time.back(), now_s() - t0, riters);
  }
  double total = now_s() - t0;
  printf("{\"mode\":\"celf\",\"n\":%d,\"m\":%lld,\"p\":%d,\"k\":%d,\"d\":%d,\"rb\":%d,\"accept\":%g,\"slack\":%g,\"threads\":%d,\"set\":",
         G.n, G.nnz() / 2, a.p, a.k, a.d, a.RB, a.accept, a.slack, S.threads);
  print_set(sel);
  printf(",\"time\":%.3f,\"t_init\":%.3f,\"init_pcg_iters\":%lld,\"refreshed\":%lld,\"refresh_batches\":%lld,"
         "\"refresh_pcg_iters\":%lld,\"spmv_cols\":%lld,\"rebuilds\":%lld,\"jl_time\":%.3f,\"round_refresh\":[",
         total, t_init, st.iters, refreshed, batches, riters, S.spmv_cols, rebuilds, st.time);
  for (size_t i = 0; i < round_refresh.size(); i++) printf("%s%d", i ? "," : "", round_refresh[i]);
  printf("],\"round_time\":[");
  for (size_t i = 0; i < round_time.size(); i++) printf("%s%.3f", i ? "," : "", round_time[i]);
  printf("],\"gains\":[");
  for (size_t i = 0; i < gains.size(); i++) printf("%s%.9g", i ? "," : "", gains[i]);
  printf("]}\n");
}

static void mode_jlg(const Graph& G, const Args& a) {
  double t0 = now_s();
  Solver S(G);
  S.ground(a.seedv);
  setup_ac(S, a.ac, a.rng ^ 0xac0ULL);
  vector<int> sel{a.seedv};
  JLStats st;
  vector<double> round_time;
  for (int round = 1; round < a.k; round++) {
    double tr = now_s();
    auto R = jl_moments(S, a.p + 1, a.d, a.BS, a.tol, a.rng + 7919ULL * round, st);
    int best = -1; double bv = -1;
    vector<double> rv(a.p + 2);
    for (int i = 0; i < G.n; i++) {
      if (S.gr[i]) continue;
      for (int j = 1; j <= a.p + 1; j++) rv[j] = R[j][i];
      double q = marginal_from_moments(rv.data(), a.p);
      if (q > bv) { bv = q; best = i; }
    }
    S.ground(best); sel.push_back(best);
    round_time.push_back(now_s() - tr);
    fprintf(stderr, "round %d: v=%d est=%.6g time=%.2f total=%.2f\n", round, best, bv, round_time.back(), now_s() - t0);
  }
  printf("{\"mode\":\"jlg\",\"n\":%d,\"m\":%lld,\"p\":%d,\"k\":%d,\"d\":%d,\"threads\":%d,\"set\":", G.n, G.nnz() / 2,
         a.p, a.k, a.d, S.threads);
  print_set(sel);
  printf(",\"time\":%.3f,\"pcg_iters\":%lld,\"spmv_cols\":%lld,\"round_time\":[", now_s() - t0, st.iters, S.spmv_cols);
  for (size_t i = 0; i < round_time.size(); i++) printf("%s%.3f", i ? "," : "", round_time[i]);
  printf("]}\n");
}

static void mode_exactmarg(const Graph& G, const Args& a) {
  Solver S(G);
  S.ground(a.seedv);
  setup_ac(S, a.ac, a.rng ^ 0xac0ULL);
  ll it = 0;
  for (int c0 = 0; c0 < G.n; c0 += a.RB) {
    vector<int> cand;
    for (int v = c0; v < min(G.n, c0 + a.RB); v++) if (!S.gr[v]) cand.push_back(v);
    if (cand.empty()) continue;
    auto val = refresh(S, cand, a.p, a.rtol, it);
    for (size_t c = 0; c < cand.size(); c++) printf("%d %.15g\n", cand[c], val[c]);
  }
}

// Evaluation on a set: Hutchinson with common Rademacher probes (same rng for every set) and lambda_min
// by Lanczos on X^{-1} with full reorthogonalization.
static void mode_eval(const Graph& G, const Args& a) {
  double t0 = now_s();
  vector<int> set;
  {
    ifstream in(a.setfile);
    int v;
    while (in >> v) set.push_back(v);
  }
  Solver S(G);
  for (int v : set) S.ground(v);
  setup_ac(S, a.ac, a.rng ^ 0xac0ULL);
  int n = G.n;
  double t1 = 0, t2 = 0, t1sq = 0;
  vector<double> pr1, pr2;
  ll iters = 0;
  // --probes <= 0: exact traces from unit vectors e_i over all active vertices.
  bool exacttr = a.probes <= 0;
  vector<int> act;
  if (exacttr) for (int i = 0; i < n; i++) if (!S.gr[i]) act.push_back(i);
  int NP = exacttr ? (int)act.size() : a.probes;
  for (int c0 = 0; c0 < NP; c0 += a.BS) {
    int B = min(a.BS, NP - c0);
    size_t N = (size_t)n * B;
    vector<double> b(N, 0.0), x(N);
    if (exacttr) {
      for (int c = 0; c < B; c++) b[(size_t)act[c0 + c] * B + c] = 1.0;
    } else {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
      for (int c = 0; c < B; c++) b[(size_t)i * B + c] = S.gr[i] ? 0.0 : rsign(a.rng ^ 0xe7a1ULL, c0 + c, i);
    }
    iters += S.solve(B, b.data(), x.data(), a.rtol);
    vector<double> s1(B, 0), s2(B, 0);
    for (int i = 0; i < n; i++)
      for (int c = 0; c < B; c++) {
        size_t q = (size_t)i * B + c;
        s1[c] += b[q] * x[q]; s2[c] += x[q] * x[q];
      }
    for (int c = 0; c < B; c++) { t1 += s1[c]; t2 += s2[c]; t1sq += s1[c] * s1[c]; pr1.push_back(s1[c]); pr2.push_back(s2[c]); }
  }
  if (!a.dumpfile.empty()) {  // per-probe values for paired comparisons across sets (same probes)
    FILE* fo = fopen(a.dumpfile.c_str(), "w");
    for (size_t c = 0; c < pr1.size(); c++) fprintf(fo, "%.17g %.17g\n", pr1[c], pr2[c]);
    fclose(fo);
  }
  double se1 = 0;
  if (!exacttr) {
    t1 /= a.probes; t2 /= a.probes;
    se1 = sqrt(max(0.0, t1sq / a.probes - t1 * t1) / a.probes);
  }
  // Lanczos on X^{-1}
  int K = a.lanczos;
  vector<vector<double>> V;
  vector<double> al, be;
  vector<double> v(n), w(n);
  {
    double s = 0;
    for (int i = 0; i < n; i++) { v[i] = S.gr[i] ? 0 : 1.0 + 0.1 * rsign(77, 0, i); s += v[i] * v[i]; }
    s = sqrt(s);
    for (int i = 0; i < n; i++) v[i] /= s;
  }
  double theta = 0, resid = 1;
  int steps = 0;
  for (int j = 0; j < K; j++) {
    V.push_back(v);
    iters += S.solve(1, v.data(), w.data(), a.rtol * 1e-2);
    double aj = 0;
    for (int i = 0; i < n; i++) aj += v[i] * w[i];
    al.push_back(aj);
    for (auto& u : V) {  // full reorthogonalization (twice)
      for (int rep = 0; rep < 2; rep++) {
        double h = 0;
        for (int i = 0; i < n; i++) h += u[i] * w[i];
        for (int i = 0; i < n; i++) w[i] -= h * u[i];
      }
    }
    double bj = 0;
    for (int i = 0; i < n; i++) bj += w[i] * w[i];
    bj = sqrt(bj);
    steps = j + 1;
    // largest Ritz value of the tridiagonal (bisection-free: dense symmetric eig via Jacobi on small matrix)
    int m = steps;
    vector<double> T(m * m, 0.0), Q(m * m, 0.0);
    for (int i = 0; i < m; i++) { T[i * m + i] = al[i]; Q[i * m + i] = 1; if (i + 1 < m) T[i * m + i + 1] = T[(i + 1) * m + i] = be[i]; }
    for (int sweep = 0; sweep < 100; sweep++) {
      double off = 0;
      for (int p = 0; p < m; p++) for (int q = p + 1; q < m; q++) off += T[p * m + q] * T[p * m + q];
      if (off < 1e-30) break;
      for (int p = 0; p < m; p++)
        for (int q = p + 1; q < m; q++) {
          double apq = T[p * m + q];
          if (fabs(apq) < 1e-300) continue;
          double tau = (T[q * m + q] - T[p * m + p]) / (2 * apq);
          double t = (tau >= 0 ? 1 : -1) / (fabs(tau) + sqrt(1 + tau * tau));
          double c = 1 / sqrt(1 + t * t), s = t * c;
          for (int r = 0; r < m; r++) {
            double trp = T[r * m + p], trq = T[r * m + q];
            T[r * m + p] = c * trp - s * trq; T[r * m + q] = s * trp + c * trq;
          }
          for (int r = 0; r < m; r++) {
            double tpr = T[p * m + r], tqr = T[q * m + r];
            T[p * m + r] = c * tpr - s * tqr; T[q * m + r] = s * tpr + c * tqr;
          }
          for (int r = 0; r < m; r++) {
            double qrp = Q[r * m + p], qrq = Q[r * m + q];
            Q[r * m + p] = c * qrp - s * qrq; Q[r * m + q] = s * qrp + c * qrq;
          }
        }
    }
    int bi = 0;
    for (int i = 1; i < m; i++) if (T[i * m + i] > T[bi * m + bi]) bi = i;
    theta = T[bi * m + bi];
    resid = fabs(bj * Q[(m - 1) * m + bi]);
    if (resid < 1e-7 * theta || bj < 1e-14) break;
    be.push_back(bj);
    for (int i = 0; i < n; i++) v[i] = w[i] / bj;
  }
  printf("{\"mode\":\"eval\",\"n\":%d,\"k\":%zu,\"tr_inv\":%.10g,\"tr_inv_se\":%.4g,\"tr_inv2\":%.10g,\"cfcc\":%.10g,"
         "\"lambda_min\":%.10g,\"lanczos_steps\":%d,\"lanczos_rel_resid\":%.3g,\"probes\":%d,\"pcg_iters\":%lld,\"time\":%.3f}\n",
         n, set.size(), t1, se1, t2, n / t1, 1.0 / theta, steps, resid / theta, NP * (exacttr ? -1 : 1), iters, now_s() - t0);
}

// Exact dense greedy for validation (n up to a few thousand).
static void mode_dense(const Graph& G, const Args& a) {
  int n = G.n;
  vector<int> alive;
  vector<char> gr(n, 0);
  gr[a.seedv] = 1;
  vector<int> sel{a.seedv};
  for (int round = 1; round < a.k; round++) {
    alive.clear();
    for (int i = 0; i < n; i++) if (!gr[i]) alive.push_back(i);
    int d = (int)alive.size();
    vector<int> pos(n, -1);
    for (int i = 0; i < d; i++) pos[alive[i]] = i;
    // A = X^{-1} by Cholesky + inversion of columns; then powers of A applied to e_v -> moments
    vector<double> X((size_t)d * d, 0.0);
    for (int i = 0; i < d; i++) {
      int u = alive[i];
      X[(size_t)i * d + i] = G.deg[u];
      for (ll e = G.ptr[u]; e < G.ptr[u + 1]; e++) if (pos[G.adj[e]] >= 0) X[(size_t)i * d + pos[G.adj[e]]] -= 1;
    }
    // Cholesky
    for (int j = 0; j < d; j++) {
      double s = X[(size_t)j * d + j];
      for (int t = 0; t < j; t++) s -= X[(size_t)j * d + t] * X[(size_t)j * d + t];
      s = sqrt(s);
      X[(size_t)j * d + j] = s;
#pragma omp parallel for
      for (int i = j + 1; i < d; i++) {
        double v = X[(size_t)i * d + j];
        for (int t = 0; t < j; t++) v -= X[(size_t)i * d + t] * X[(size_t)j * d + t];
        X[(size_t)i * d + j] = v / s;
      }
    }
    // A = X^{-1}: solve for all columns
    vector<double> A((size_t)d * d, 0.0);
#pragma omp parallel for
    for (int c = 0; c < d; c++) {
      vector<double> y(d, 0.0);
      y[c] = 1;
      for (int i = 0; i < d; i++) { double v = y[i]; for (int t = 0; t < i; t++) v -= X[(size_t)i * d + t] * y[t]; y[i] = v / X[(size_t)i * d + i]; }
      for (int i = d - 1; i >= 0; i--) { double v = y[i]; for (int t = i + 1; t < d; t++) v -= X[(size_t)t * d + i] * y[t]; y[i] = v / X[(size_t)i * d + i]; }
      for (int i = 0; i < d; i++) A[(size_t)i * d + c] = y[i];
    }
    int best = -1; double bv = -1;
    int len = (a.p + 2) / 2;
#pragma omp parallel
    {
      int lb = -1; double lv = -1;
      vector<double> y0(d), y1(d), r(2 * len + 2);
#pragma omp for
      for (int c = 0; c < d; c++) {
        fill(y0.begin(), y0.end(), 0.0); y0[c] = 1;
        for (int st = 1; st <= len; st++) {
          for (int i = 0; i < d; i++) { double s = 0; for (int t = 0; t < d; t++) s += A[(size_t)i * d + t] * y0[t]; y1[i] = s; }
          double sq = 0, dt = 0;
          for (int i = 0; i < d; i++) { sq += y1[i] * y1[i]; dt += y0[i] * y1[i]; }
          r[2 * st - 1] = dt; r[2 * st] = sq;
          swap(y0, y1);
        }
        double q = marginal_from_moments(r.data(), a.p);
        if (q > lv || (q == lv && alive[c] < (lb >= 0 ? alive[lb] : INT_MAX))) { lv = q; lb = c; }
      }
#pragma omp critical
      if (lv > bv || (lv == bv && alive[lb] < alive[best])) { bv = lv; best = lb; }
    }
    gr[alive[best]] = 1;
    sel.push_back(alive[best]);
  }
  printf("{\"mode\":\"dense\",\"n\":%d,\"p\":%d,\"k\":%d,\"set\":", n, a.p, a.k);
  print_set(sel);
  printf("}\n");
}



// ---------------------------------------------------------------- incremental lazy greedy
// JL moments at S0; after each selection v every cached moment r_j(u), j <= p+1, is updated exactly with the
// chain a_i = X^{-(i+1)} e_v (i = 0..p) of the pre-grounding matrix:
//   m_j(u) = r_j(u) - c sum_{i<j} gamma_{j-1-i} a_i(u),  gamma_t = a_t(u) - c sum_{i<t} gamma_{t-1-i} s_i,
//   c = 1 / x_v, s_i = <a_0, a_i>.
// Keys are the marginals of the cached moments; the heap top is certified by exact refresh (whose moments
// then replace the cache entries).
static void mode_inc(const Graph& G, const Args& a) {
  double t0 = now_s();
  int n = G.n, p = a.p;
  Solver S(G);
  S.ground(a.seedv);
  setup_ac(S, a.ac, a.rng ^ 0xac0ULL);
  vector<int> sel{a.seedv};
  JLStats st;
  vector<double> Ysk[2];
  if (a.sketch && p != 1) { fprintf(stderr, "--sketch needs p=1\n"); exit(1); }
  auto R = jl_moments(S, p + 1, a.d, a.BS, a.tol, a.rng, st, a.sketch ? Ysk : nullptr);
  if (a.exactinit) {  // validation: exact initial moments
    ll it = 0;
    for (int c0 = 0; c0 < n; c0 += 64) {
      vector<int> cand;
      for (int u = c0; u < min(n, c0 + 64); u++) if (!S.gr[u]) cand.push_back(u);
      vector<vector<double>> mm;
      refresh(S, cand, p, 1e-11, it, &mm);
      for (size_t c = 0; c < cand.size(); c++) for (int j = 1; j <= p + 1; j++) R[j][cand[c]] = mm[c][j];
    }
  }
  double t_init = now_s() - t0;
  fprintf(stderr, "init: %.2f s, pcg iters %lld\n", t_init, st.iters);
  ll refreshed = 0, riters = 0, uiters = 0;
  vector<double> round_time, gains;
  vector<int> round_refresh;
  vector<double> rv(p + 2);
  auto keyof = [&](int u) {
    for (int j = 1; j <= p + 1; j++) rv[j] = R[j][u];
    return marginal_from_moments(rv.data(), p);
  };
  double t_update = 0;
  vector<char> exact(n, (char)a.exactinit);
  int budget_hits = 0;  // rounds stopped by --budget: pick = best exactly refreshed candidate (dominates the JL argmax, which is refreshed first)
  // Absolute error radii of the cached moments: zeta * initial estimate; exact updates subtract exact
  // quantities, so the radius of a cache entry never changes (theory.tex, cache-error identity).
  vector<vector<double>> E(p + 2);
  if (a.keyub == 2) for (int j = 1; j <= p + 1; j++) { E[j].resize(n); for (int u = 0; u < n; u++) E[j][u] = a.zeta * R[j][u]; }
  // Certified upper key from moment intervals: tighter endpoint bound U2 (theory.tex, prop:tightendpoint).
  auto certified_key = [&](int u) {
    double lo1 = max(R[1][u] - E[1][u], 1.0 / G.deg[u]), hi1 = R[1][u] + E[1][u];
    auto lo = [&](int j) { return max(R[j][u] - E[j][u], pow(lo1, j)); };   // Jensen: r_j >= r_1^j
    auto hi = [&](int j) { return R[j][u] + E[j][u]; };
    // a_j = r_{j+1} / r_1
    double up = hi(p + 1) / lo1;
    if (p == 1) return up;
    double la1 = lo(2) / hi1, ua1 = hi(2) / lo1, la2 = lo(3) / hi1;
    double vm = max(0.0, la2 - ua1 * ua1);
    double U = p * up - (p - 1) * pow(la1, p) - (p >= 3 ? p * (p - 2) * vm * pow(la1, p - 2) : 0.0);
    return U;
  };
  for (int round = 1; round < a.k; round++) {
    double tr = now_s();
    vector<pair<double, int>> H;
    H.reserve(n);
    for (int u = 0; u < n; u++) {
      if (S.gr[u]) continue;
      double key;
      if (exact[u]) key = keyof(u);
      else if (a.keyub == 2) key = certified_key(u);
      else if (a.keyub) key = p * R[p + 1][u] / R[1][u] * (1 + a.slack);  // q_p <= p r_{p+1} / r_1 (T7)
      else key = keyof(u) * (1 + a.slack);
      H.push_back({key, u});
    }
    make_heap(H.begin(), H.end());
    int rr = 0, best = -1, chain_owner = -1;
    double bv = -1;
    vector<vector<double>> best_chain;
    while (true) {
      // exact cache entries at the heap top are certified values: no solve needed
      while (!H.empty() && exact[H.front().second]) {
        if (H.front().first > bv) { bv = H.front().first; best = H.front().second; }
        pop_heap(H.begin(), H.end()); H.pop_back();
      }
      double K = H.empty() ? -1.0 : H.front().first;
      if (best >= 0 && bv >= (1.0 - a.accept) * K) break;
      if (best >= 0 && a.budget > 0 && rr >= a.budget) { budget_hits++; break; }
      vector<int> cand;
      while (!H.empty() && (int)cand.size() < a.RB) {
        if (!exact[H.front().second]) cand.push_back(H.front().second);
        else if (H.front().first > bv) { bv = H.front().first; best = H.front().second; }
        pop_heap(H.begin(), H.end()); H.pop_back();
      }
      if (cand.empty()) continue;
      vector<vector<double>> mm, ch;
      auto val = refresh(S, cand, p, a.rtol, riters, &mm, &ch);
      {
        int cb = (int)(max_element(val.begin(), val.end()) - val.begin());
        if (val[cb] > bv) { best_chain.swap(ch); chain_owner = cand[cb]; }
      }
      for (size_t c = 0; c < cand.size(); c++) {
        for (int j = 1; j <= p + 1; j++) { R[j][cand[c]] = mm[c][j]; if (a.keyub == 2) E[j][cand[c]] = 0; }
        exact[cand[c]] = 1;
        if (val[c] > bv) { bv = val[c]; best = cand[c]; }
      }
      rr += cand.size(); refreshed += cand.size();
      if (getenv("GDEBUG")) fprintf(stderr, "  batch rr=%d K=%.6g bv=%.6g iters=%lld t=%.2f\n", rr, K, bv, riters, now_s() - tr);
    }
    // exact update of all caches by the chain from e_best on the current matrix
    double tu = now_s();
    if (round + 1 < a.k) {
      vector<vector<double>> A(p + 1, vector<double>(n));
      vector<double> b(n, 0.0);
      b[best] = 1;
      int have = 0;
      if (chain_owner == best && a.reuse) {  // y_1..y_len from the certifying refresh
        have = min((int)best_chain.size(), p + 1);
        for (int i = 0; i < have; i++) A[i] = best_chain[i];
        if (have > 0) b = A[have - 1];
      }
      int need = (a.sketch && !a.keepexact) ? 0 : p;  // sketch-only mode needs only x = X^{-1} e_best
      for (int i = have; i <= need; i++) { uiters += S.solve(1, b.data(), A[i].data(), a.itol); b = A[i]; }
      double xv = A[0][best], c = 1.0 / xv;
      if (a.sketch) {
        // Y' = Y - x Y[v,:] / x_v for both sketches (Sherman-Morrison on the grounded inverse); keys from the new sketches
        int d = a.d;
        const double* x = A[0].data();
        vector<double> yv[2];
        for (int f = 0; f < 2; f++) yv[f].assign(Ysk[f].begin() + (size_t)best * d, Ysk[f].begin() + (size_t)(best + 1) * d);
#pragma omp parallel for schedule(static)
        for (int u = 0; u < n; u++) {
          if (S.gr[u]) continue;
          double cu = c * x[u], s0 = 0, s1 = 0;
          double* y0 = Ysk[0].data() + (size_t)u * d; double* y1 = Ysk[1].data() + (size_t)u * d;
          for (int q = 0; q < d; q++) { y0[q] -= cu * yv[0][q]; y1[q] -= cu * yv[1][q]; s0 += y0[q] * y0[q]; s1 += y1[q] * y1[q]; }
          if (!(a.keepexact && exact[u])) { R[1][u] = s0 / d; R[2][u] = s1 / d; exact[u] = 0; }
        }
        for (int f = 0; f < 2; f++) for (int q = 0; q < d; q++) Ysk[f][(size_t)best * d + q] = 0;
      }
      if (!a.sketch || a.keepexact) {  // exact scalar update (sketch + keepexact: only refreshed entries, which stay exact)
      vector<double> sv(p + 1, 0.0);
      for (int i = 0; i <= p; i++) { double t = 0; for (int u = 0; u < n; u++) t += A[0][u] * A[i][u]; sv[i] = t; }
#pragma omp parallel
      {
        vector<double> gam(p + 2);
#pragma omp for schedule(static)
        for (int u = 0; u < n; u++) {
          if (S.gr[u] || u == best) continue;
          if (a.sketch && !exact[u]) continue;
          for (int t = 0; t <= p; t++) {
            double g = A[t][u];
            for (int i = 0; i < t; i++) g -= c * gam[t - 1 - i] * sv[i];
            gam[t] = g;
          }
          for (int j = 1; j <= p + 1; j++) {
            double m = 0;
            for (int i = 0; i < j; i++) m += gam[j - 1 - i] * A[i][u];
            R[j][u] -= c * m;
          }
        }
      }
      }
    }
    t_update += now_s() - tu;
    S.ground(best); sel.push_back(best); gains.push_back(bv);
    if (a.sketch && getenv("GSKCHECK") && round + 1 < a.k) {  // maintained sketch == fresh sketch with the same randomness
      JLStats st2;
      auto F = jl_moments(S, 2, a.d, a.BS, 1e-10, a.rng, st2);
      double w1 = 0, w2 = 0;
      for (int u = 0; u < n; u++) if (!S.gr[u]) { w1 = max(w1, fabs(F[1][u] - R[1][u]) / F[1][u]); w2 = max(w2, fabs(F[2][u] - R[2][u]) / F[2][u]); }
      fprintf(stderr, "  sketch check: max rel diff r1 %.2e r2 %.2e\n", w1, w2);
    }
    if (a.checkupd && round + 1 < a.k) {
      vector<int> cand;
      for (int u = 0; u < n && (int)cand.size() < 16; u += 1 + n / 17) if (!S.gr[u]) cand.push_back(u);
      vector<vector<double>> mm;
      ll dummy = 0;
      refresh(S, cand, p, 1e-11, dummy, &mm);
      double worst = 0;
      vector<double> wj(p + 2, 0.0);
      for (size_t c = 0; c < cand.size(); c++)
        for (int j = 1; j <= p + 1; j++) {
          double e = fabs(R[j][cand[c]] - mm[c][j]) / fabs(mm[c][j]);
          wj[j] = max(wj[j], e);
          worst = max(worst, e);
        }
      fprintf(stderr, "checkupd round %d: max rel moment error %.3e  per j:", round, worst);
      for (int j = 1; j <= p + 1; j++) fprintf(stderr, " %.1e", wj[j]);
      fprintf(stderr, "\n");
    }
    round_time.push_back(now_s() - tr);
    round_refresh.push_back(rr);
    fprintf(stderr, "inc round %d: v=%d gain=%.6g refreshed=%d time=%.2f total=%.2f\n", round, best, bv, rr,
            round_time.back(), now_s() - t0);
  }
  printf("{\"mode\":\"inc\",\"n\":%d,\"m\":%lld,\"p\":%d,\"k\":%d,\"d\":%d,\"rb\":%d,\"slack\":%g,\"accept\":%g,\"threads\":%d,\"set\":",
         n, G.nnz() / 2, p, a.k, a.d, a.RB, a.slack, a.accept, S.threads);
  print_set(sel);
  printf(",\"time\":%.3f,\"t_init\":%.3f,\"t_update\":%.3f,\"init_pcg_iters\":%lld,\"refreshed\":%lld,\"refresh_pcg_iters\":%lld,"
         "\"update_pcg_iters\":%lld,\"spmv_cols\":%lld,\"budget\":%d,\"budget_hits\":%d,\"round_refresh\":[",
         now_s() - t0, t_init, t_update, st.iters, refreshed, riters, uiters, S.spmv_cols, a.budget, budget_hits);
  for (size_t i = 0; i < round_refresh.size(); i++) printf("%s%d", i ? "," : "", round_refresh[i]);
  printf("],\"round_time\":[");
  for (size_t i = 0; i < round_time.size(); i++) printf("%s%.3f", i ? "," : "", round_time[i]);
  printf("],\"gains\":[");
  for (size_t i = 0; i < gains.size(); i++) printf("%s%.9g", i ? "," : "", gains[i]);
  printf("]}\n");
}


// ---------------------------------------------------------------- Zhou et al. "Fast" for max lambda_min
// Each round: eigenvector u of lambda_min(X_S) by inverse power iteration with PCG (warm start), score
// 2 u_j sum_{i in N_j \ S} u_i, pick the maximum.
static void mode_zhou(const Graph& G, const Args& a) {
  double t0 = now_s();
  int n = G.n;
  Solver S(G);
  S.ground(a.seedv);
  setup_ac(S, a.ac, a.rng ^ 0xac0ULL);
  vector<int> sel{a.seedv};
  vector<double> u(n, 1.0), v(n), Xv(n);
  ll iters = 0, pits = 0;
  for (int round = 1; round < a.k; round++) {
    double tr = now_s();
    for (int i = 0; i < n; i++) if (S.gr[i]) u[i] = 0;
    double nu = 0;
    for (int i = 0; i < n; i++) nu += u[i] * u[i];
    for (int i = 0; i < n; i++) u[i] /= sqrt(nu);
    double rho_old = -1, rho = 0;
    for (int it = 0; it < 200; it++) {
      iters += S.solve(1, u.data(), v.data(), a.rtol);
      pits++;
      double nv = 0;
      for (int i = 0; i < n; i++) nv += v[i] * v[i];
      nv = sqrt(nv);
      for (int i = 0; i < n; i++) v[i] /= nv;
      double q = 0;
      S.apply(1, v.data(), Xv.data(), &q);
      rho = q;
      u = v;
      if (rho_old > 0 && fabs(rho_old - rho) <= a.eps * rho) break;
      rho_old = rho;
    }
    int best = -1; double bv = -1e300;
    for (int j = 0; j < n; j++) {
      if (S.gr[j]) continue;
      double sum = 0;
      for (ll e = G.ptr[j]; e < G.ptr[j + 1]; e++) sum += u[G.adj[e]];  // grounded entries are zero
      double sc = 2 * u[j] * sum;
      if (sc > bv) { bv = sc; best = j; }
    }
    S.ground(best); sel.push_back(best);
    fprintf(stderr, "zhou round %d: v=%d lambda=%.6g time=%.2f total=%.2f\n", round, best, rho, now_s() - tr, now_s() - t0);
  }
  printf("{\"mode\":\"zhou\",\"n\":%d,\"m\":%lld,\"k\":%d,\"threads\":%d,\"set\":", n, G.nnz() / 2, a.k, S.threads);
  print_set(sel);
  printf(",\"time\":%.3f,\"power_iters\":%lld,\"pcg_iters\":%lld}\n", now_s() - t0, pits, iters);
}

// ---------------------------------------------------------------- ForestCFCM baseline (Xia--Zhang, Alg. 1-3)
// Per round: sample spanning forests rooted at S by Wilson's algorithm; per forest, edge flows of the JL
// sources W (w Rademacher rows / sqrt(w)) are subtree sums; voltages Phi_{w_j,S}(u) accumulate along the fixed
// BFS tree from S (Lemma 3.3); z_u = Phi_{u,S}(u) uses the same BFS path with forest-ancestor tests.
// Delta'(u) = ||Y e_u||^2 / z_u. Forest count: fixed (--forests) or doubling with an empirical-Bernstein stop
// on the selected candidate's denominator (--forests 0, --eps).

static void mode_forest(const Graph& G, const Args& a) {
  double t0 = now_s();
  int n = G.n, w = a.d;
  vector<char> gr(n, 0);
  gr[a.seedv] = 1;
  vector<int> sel{a.seedv};
  int T = omp_get_max_threads();
  ll total_forests = 0, total_steps = 0;
  vector<double> round_time;
  vector<ll> round_forests;
  for (int round = 1; round < a.k; round++) {
    double tr = now_s();
    // BFS tree from S
    vector<int> bp(n, -1), bfs, dep(n, 0);
    for (int i = 0; i < n; i++) if (gr[i]) { bfs.push_back(i); bp[i] = i; }
    for (size_t h = 0; h < bfs.size(); h++) {
      int x = bfs[h];
      for (ll e = G.ptr[x]; e < G.ptr[x + 1]; e++) {
        int y = G.adj[e];
        if (bp[y] < 0) { bp[y] = x; dep[y] = dep[x] + 1; bfs.push_back(y); }
      }
    }
    int maxdep = 0;
    for (int i = 0; i < n; i++) maxdep = max(maxdep, dep[i]);
    // JL source matrix W (w x n), entries +-1/sqrt(w), on V \ S
    auto Wv = [&](int j, int v) { return rsign(a.rng ^ 0xf0e57ULL, (uint64_t)(round * 1000003 + j), (uint64_t)v) / sqrt((double)w); };
    vector<double> Wm((size_t)n * w);
#pragma omp parallel for
    for (int v = 0; v < n; v++) for (int j = 0; j < w; j++) Wm[(size_t)v * w + j] = gr[v] ? 0.0 : Wv(j, v);
    vector<double> Y((size_t)n * w, 0.0), Z(n, 0.0), Z2(n, 0.0);
    ll forests = 0;
    int best = -1;
    int target = a.forests > 0 ? a.forests : 0;
    for (int batch = 0;; batch++) {
      int nb = a.forests > 0 ? a.forests : (1 << batch) * T;
      if (a.forests > 0 && batch > 0) break;
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        Rng rng{a.rng * 7777 + (uint64_t)round * 1315423911ULL + (uint64_t)(forests + tid) * 2654435761ULL};
        vector<double> Yl((size_t)n * w, 0.0), Zl(n, 0.0), Z2l(n, 0.0);
        vector<int> nxt(n), order, chain, tin(n), tout(n), cstart(n + 1), clist(n), st;
        vector<char> inF(n);
        vector<double> f((size_t)n * w), phi((size_t)n * w);
        ll steps = 0;
#pragma omp for schedule(dynamic, 1)
        for (int fi = 0; fi < nb; fi++) {
          // Wilson rooted at S
          for (int i = 0; i < n; i++) inF[i] = gr[i];
          order.clear();
          for (int u = 0; u < n; u++) {
            if (inF[u]) continue;
            int i = u;
            while (!inF[i]) {
              ll dg = G.ptr[i + 1] - G.ptr[i];
              nxt[i] = G.adj[G.ptr[i] + (ll)(rng.next() % (uint64_t)dg)];
              i = nxt[i]; steps++;
            }
            chain.clear();
            i = u;
            while (!inF[i]) { inF[i] = 1; chain.push_back(i); i = nxt[i]; }
            for (int c = (int)chain.size() - 1; c >= 0; c--) order.push_back(chain[c]);  // parents first
          }
          // edge flows of W sources: subtree sums, children before parents
          memcpy(f.data(), Wm.data(), sizeof(double) * (size_t)n * w);
          for (int c = (int)order.size() - 1; c >= 0; c--) {
            int x = order[c], px = nxt[x];
            if (gr[px]) continue;
            double* fp = f.data() + (size_t)px * w;
            const double* fx = f.data() + (size_t)x * w;
            for (int j = 0; j < w; j++) fp[j] += fx[j];
          }
          // voltages along the BFS tree from S
          for (int x : bfs) {
            double* ph = phi.data() + (size_t)x * w;
            if (gr[x]) { for (int j = 0; j < w; j++) ph[j] = 0; continue; }
            int b = bp[x];
            const double* pb = phi.data() + (size_t)b * w;
            if (nxt[x] == b) { const double* fx = f.data() + (size_t)x * w; for (int j = 0; j < w; j++) ph[j] = pb[j] + fx[j]; }
            else if (!gr[b] && nxt[b] == x) { const double* fb = f.data() + (size_t)b * w; for (int j = 0; j < w; j++) ph[j] = pb[j] - fb[j]; }
            else for (int j = 0; j < w; j++) ph[j] = pb[j];
          }
          for (size_t q = 0; q < (size_t)n * w; q++) Yl[q] += phi[q];
          // forest Euler intervals for ancestor tests
          fill(cstart.begin(), cstart.end(), 0);
          for (int x : order) cstart[nxt[x] + 1]++;
          for (int i = 0; i < n; i++) cstart[i + 1] += cstart[i];
          {
            vector<int> pos(cstart.begin(), cstart.end() - 1);
            for (int x : order) clist[pos[nxt[x]]++] = x;
          }
          int clock = 0;
          for (int r = 0; r < n; r++) {
            if (!gr[r]) continue;
            st.clear(); st.push_back(r); tin[r] = clock++;
            vector<int> it;  // child iterator per stack level
            it.push_back(cstart[r]);
            while (!st.empty()) {
              int x = st.back();
              if (it.back() < cstart[x + 1]) {
                int y = clist[it.back()++];
                tin[y] = clock++; st.push_back(y); it.push_back(cstart[y]);
              } else { tout[x] = clock++; st.pop_back(); it.pop_back(); }
            }
          }
          auto anc = [&](int x, int u) { return tin[x] <= tin[u] && tout[u] <= tout[x]; };
          for (int u = 0; u < n; u++) {
            if (gr[u]) continue;
            double zu = 0;
            for (int x = u; !gr[x]; x = bp[x]) {
              int b = bp[x];
              if (nxt[x] == b && anc(x, u)) zu += 1;
              else if (!gr[b] && nxt[b] == x && anc(b, u)) zu -= 1;
            }
            Zl[u] += zu; Z2l[u] += zu * zu;
          }
        }
#pragma omp critical
        {
          for (size_t q = 0; q < (size_t)n * w; q++) Y[q] += Yl[q];
          for (int i = 0; i < n; i++) { Z[i] += Zl[i]; Z2[i] += Z2l[i]; }
          total_steps += steps;
        }
      }
      forests += nb;
      // marginal estimates and selection
      double bv = -1;
      best = -1;
      for (int u = 0; u < n; u++) {
        if (gr[u]) continue;
        double zu = Z[u] / forests;
        if (zu <= 0) continue;
        double s2 = 0;
        for (int j = 0; j < w; j++) { double y = Y[(size_t)u * w + j] / forests; s2 += y * y; }
        double d = s2 / zu;
        if (d > bv) { bv = d; best = u; }
      }
      if (a.forests > 0) break;
      // empirical Bernstein (Lemma 3.6) on z of the selected candidate, X_sup = BFS depth bound
      double mean = Z[best] / forests, var = max(0.0, Z2[best] / forests - mean * mean);
      double delta = 1.0 / n, lg = log(3.0 / delta);
      double err = sqrt(2 * var * lg / forests) + 3.0 * maxdep * lg / forests;
      if (err <= a.eps * (mean - err) || forests >= a.maxforests) break;
    }
    (void)target;
    gr[best] = 1;
    sel.push_back(best);
    total_forests += forests;
    round_time.push_back(now_s() - tr);
    round_forests.push_back(forests);
    fprintf(stderr, "forest round %d: v=%d forests=%lld time=%.2f total=%.2f\n", round, best, forests, round_time.back(), now_s() - t0);
  }
  printf("{\"mode\":\"forest\",\"n\":%d,\"m\":%lld,\"k\":%d,\"w\":%d,\"threads\":%d,\"set\":", n, G.nnz() / 2, a.k, w, T);
  print_set(sel);
  printf(",\"time\":%.3f,\"forests\":%lld,\"walk_steps\":%lld,\"round_forests\":[", now_s() - t0, total_forests, total_steps);
  for (size_t i = 0; i < round_forests.size(); i++) printf("%s%lld", i ? "," : "", round_forests[i]);
  printf("],\"round_time\":[");
  for (size_t i = 0; i < round_time.size(); i++) printf("%s%.3f", i ? "," : "", round_time[i]);
  printf("]}\n");
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ground <mode> <graph> [options]\n"); return 1; }
  if (string(argv[1]) == "convert") { convert(argv[2], argv[3]); return 0; }
  Args a = parse(argc, argv);
  if (a.threads > 0) omp_set_num_threads(a.threads);
  Graph G = load(a.graph);
  if (a.mode == "seed") mode_seed(G, a);
  else if (a.mode == "celf") mode_celf(G, a);
  else if (a.mode == "jlg") mode_jlg(G, a);
  else if (a.mode == "eval") mode_eval(G, a);
  else if (a.mode == "dense") mode_dense(G, a);
  else if (a.mode == "forest") mode_forest(G, a);
  else if (a.mode == "inc") mode_inc(G, a);
  else if (a.mode == "zhou") mode_zhou(G, a);
  else if (a.mode == "exactmarg") mode_exactmarg(G, a);
  else { fprintf(stderr, "unknown mode\n"); return 1; }
  return 0;
}

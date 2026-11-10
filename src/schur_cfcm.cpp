// SchurCFCM reimplementation of Xia--Zhang (arXiv:2504.04472), Algorithms 1, 4, 5.
// Independent of ground.cpp. See notes/schur_impl.md for practical stopping and symmetrization.
// Build: g++ -O3 -march=native -std=c++17 -fopenmp -o schur_cfcm src/schur_cfcm.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using ll = long long;
static double now_s() { return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count(); }
static uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
static int rsign(uint64_t a, uint64_t b, uint64_t c) { return (mix64(a ^ mix64(b ^ mix64(c))) & 1) ? 1 : -1; }
struct Rng {
  uint64_t s;
  uint64_t next() { s += 0x9e3779b97f4a7c15ULL; return mix64(s); }
  // Rejection avoids modulo bias in the uniform-neighbor distribution.
  uint64_t below(uint64_t n) { uint64_t x, threshold = -n % n; do { x = next(); } while (x < threshold); return x % n; }
};
struct Graph {
  int n = 0;
  vector<ll> ptr;
  vector<int> adj;
  int degree(int u) const { return int(ptr[u + 1] - ptr[u]); }
};
static Graph load(const string& path) {
  ifstream f(path, ios::binary);
  if (!f) throw runtime_error("cannot open graph: " + path);
  auto read = [&](void* p, size_t n) { if (!f.read((char*)p, n)) throw runtime_error("truncated GRND graph"); };
  char magic[4]; Graph g; ll nnz;
  read(magic, 4); read(&g.n, 4); read(&nnz, 8);
  if (memcmp(magic, "GRND", 4) || g.n <= 0 || nnz < 0 || (nnz & 1)) throw runtime_error("invalid GRND header");
  auto start = f.tellg(); f.seekg(0, ios::end); auto end = f.tellg(); f.seekg(start);
  if ((uint64_t)(end - start) != 8ULL * (g.n + 1ULL) + 4ULL * nnz) throw runtime_error("invalid GRND length");
  g.ptr.resize((size_t)g.n + 1); g.adj.resize(nnz);
  read(g.ptr.data(), g.ptr.size() * 8); read(g.adj.data(), g.adj.size() * 4);
  if (g.ptr.front() != 0 || g.ptr.back() != nnz) throw runtime_error("invalid CSR offsets");
  for (int u = 0; u < g.n; ++u) {
    if (g.ptr[u] < 0 || g.ptr[u + 1] < g.ptr[u] || g.ptr[u + 1] > nnz || g.ptr[u + 1] - g.ptr[u] > INT_MAX)
      throw runtime_error("invalid CSR row");
    int previous = -1;
    for (ll e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
      int v = g.adj[e];
      if (v < 0 || v >= g.n || v == u || v <= previous) throw runtime_error("GRND must have sorted simple adjacency rows");
      previous = v;
    }
  }
  return g;
}
struct BFS {
  vector<int> parent, order, depth;
  int maxdepth = 0;
  BFS(const Graph& g, const vector<char>& roots) : parent(g.n, -1), depth(g.n, 0) {
    for (int u = 0; u < g.n; ++u) if (roots[u]) { parent[u] = u; order.push_back(u); }
    for (size_t h = 0; h < order.size(); ++h) {
      int u = order[h];
      for (ll e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
        int v = g.adj[e];
        if (parent[v] < 0) { parent[v] = u; depth[v] = depth[u] + 1; maxdepth = max(maxdepth, depth[v]); order.push_back(v); }
      }
    }
    if ((int)order.size() != g.n) throw runtime_error("graph is disconnected from the supplied roots");
  }
};
struct RootChoice { vector<int> nodes; int dmax = 0; };
// Section IV-B: repeatedly remove a maximum-degree vertex of the REMAINING graph.
// Section V-A: minimize abs(c - dmax(T)); smaller c, then smaller vertex id, breaks ties.
static RootChoice choose_roots(const Graph& g, int requested) {
  vector<int> degree(g.n); vector<char> removed(g.n, 0);
  priority_queue<pair<int, int>> heap;
  for (int u = 0; u < g.n; ++u) { degree[u] = g.degree(u); heap.push({degree[u], -u}); }
  auto top = [&]() {
    while (!heap.empty() && (removed[-heap.top().second] || degree[-heap.top().second] != heap.top().first)) heap.pop();
    return heap.empty() ? 0 : heap.top().first;
  };
  RootChoice result; result.dmax = top();
  if (requested == 0) return result;
  vector<int> sequence; int best_c = 0, best_gap = top(), best_dmax = top();
  for (int c = 1; c <= g.n; ++c) {
    top(); int u = -heap.top().second; heap.pop(); removed[u] = 1; sequence.push_back(u);
    for (ll e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
      int v = g.adj[e];
      if (!removed[v]) { --degree[v]; heap.push({degree[v], -v}); }
    }
    int dm = top(), gap = abs(c - dm);
    if (gap < best_gap) { best_gap = gap; best_c = c; best_dmax = dm; }
    if (requested >= 0 && c == requested) return {sequence, dm};
    if (requested < 0 && c >= dm) break; // c-dmax is strictly increasing: only the crossing can minimize |.|.
  }
  sequence.resize(best_c); result.nodes = move(sequence); result.dmax = best_dmax; return result;
}
// Sparse row counts: no n-by-|T| allocation, no thresholding or dropped root mass.
using CountRow = vector<pair<int, uint32_t>>;
static void add_count(CountRow& row, int root, uint32_t amount) {
  auto it = lower_bound(row.begin(), row.end(), root, [](const auto& x, int y) { return x.first < y; });
  if (it != row.end() && it->first == root) it->second += amount;
  else row.insert(it, {root, amount});
}
static bool inverse_spd(const vector<double>& matrix, int n, vector<double>& inverse) {
  vector<double> lower(matrix);
  for (int j = 0; j < n; ++j) {
    double v = lower[(size_t)j*n+j];
    for (int q = 0; q < j; ++q) v -= lower[(size_t)j*n+q] * lower[(size_t)j*n+q];
    if (!(v > 0) || !isfinite(v)) return false;
    lower[(size_t)j*n+j] = sqrt(v);
    for (int i = j+1; i < n; ++i) {
      double a = lower[(size_t)i*n+j];
      for (int q = 0; q < j; ++q) a -= lower[(size_t)i*n+q] * lower[(size_t)j*n+q];
      lower[(size_t)i*n+j] = a / lower[(size_t)j*n+j];
    }
  }
  inverse.assign((size_t)n*n, 0);
#pragma omp parallel for schedule(static)
  for (int j = 0; j < n; ++j) {
    vector<double> y(n, 0); y[j] = 1;
    for (int i = 0; i < n; ++i) {
      for (int q = 0; q < i; ++q) y[i] -= lower[(size_t)i*n+q] * y[q];
      y[i] /= lower[(size_t)i*n+i];
    }
    for (int i = n-1; i >= 0; --i) {
      for (int q = i+1; q < n; ++q) y[i] -= lower[(size_t)q*n+i] * y[q];
      y[i] /= lower[(size_t)i*n+i];
    }
    for (int i = 0; i < n; ++i) inverse[(size_t)i*n+j] = y[i];
  }
  for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) {
    double v = (inverse[(size_t)i*n+j] + inverse[(size_t)j*n+i]) * .5;
    inverse[(size_t)i*n+j] = inverse[(size_t)j*n+i] = v;
  }
  return true;
}
struct Estimate {
  bool valid = false;
  int best = -1;
  vector<double> diagonal, score, schur, inverse, y;
  size_t f_nnz = 0;
};
struct ForestWorker {
  vector<int> next, order, chain, root, tin, tout, cstart, children, position, stack, iter;
  vector<char> in_forest;
  vector<int32_t> flow;
  vector<ll> edge_sum, root_sum, z;
  vector<double> z2;
  ll steps = 0;
  ForestWorker(int n, int w, int c) : next(n), root(n), tin(n), tout(n), cstart(n+1), children(n), position(n),
      in_forest(n), flow((size_t)n*w), edge_sum((size_t)n*w), root_sum((size_t)c*w), z(n), z2(n) {
    order.reserve(n); chain.reserve(n); stack.reserve(n); iter.reserve(n);
  }
  void clear_sums() {
    fill(edge_sum.begin(), edge_sum.end(), 0); fill(root_sum.begin(), root_sum.end(), 0);
    fill(z.begin(), z.end(), 0); fill(z2.begin(), z2.end(), 0);
    steps = 0;
  }
};
struct SchurSampler {
  const Graph& g;
  int n, w, c, threads, round;
  uint64_t seed;
  vector<char> selected, roots;
  vector<int> tnodes, tid;
  BFS bfs;
  int stop_depth;
  vector<int8_t> signs;
  double source_scale;
  vector<ll> edge_sum, root_sum, z;
  vector<double> z2;
  vector<CountRow> counts;
  vector<unique_ptr<ForestWorker>> workers;
  ll forests = 0, walk_steps = 0;
  SchurSampler(const Graph& graph, const vector<char>& s, vector<int> t, int dim, int nt, uint64_t rng, int r,
               bool identity_sources = false) : g(graph), n(g.n), w(dim), c(t.size()), threads(nt), round(r), seed(rng),
      selected(s), roots(s), tnodes(move(t)), tid(n, -1),
      bfs(g, [&]() { auto mask=s; for(int u:tnodes) mask[u]=1; return mask; }()),
      stop_depth(bfs.maxdepth), signs((size_t)n*w), source_scale(identity_sources ? 1.0 : 1.0/sqrt(w)),
      edge_sum((size_t)n*w), root_sum((size_t)c*w), z(n), z2(n), counts(n) {
    if (identity_sources && w != n) throw runtime_error("identity test source needs d=n");
    for (int j = 0; j < c; ++j) {
      int u = tnodes[j];
      if (u < 0 || u >= n || roots[u]) throw runtime_error("invalid or overlapping T");
      roots[u] = 1; tid[u] = j;
    }
#pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) for (int j = 0; j < w; ++j)
      signs[(size_t)u*w+j] = selected[u] ? 0 : (identity_sources ? int(u == j) : rsign(seed ^ 0xf0e57ULL, (uint64_t)round*1000003+j, u));
  }
  void sample_one(ForestWorker& a, uint64_t index) {
    Rng rng{mix64(seed ^ mix64((uint64_t)round) ^ mix64(index + 0x5363687572ULL))};
    a.in_forest = roots; a.order.clear();
    for (int u = 0; u < n; ++u) if (roots[u]) { a.next[u] = u; a.root[u] = tid[u]; }
    for (int u = 0; u < n; ++u) {
      if (a.in_forest[u]) continue;
      int x = u;
      while (!a.in_forest[x]) { a.next[x] = g.adj[g.ptr[x] + rng.below(g.degree(x))]; x = a.next[x]; ++a.steps; }
      a.chain.clear(); x = u;
      while (!a.in_forest[x]) { a.in_forest[x] = 1; a.chain.push_back(x); x = a.next[x]; }
      for (auto it = a.chain.rbegin(); it != a.chain.rend(); ++it) {
        a.order.push_back(*it); a.root[*it] = a.root[a.next[*it]];
      }
    }
    // Integer +/-1 sources make all per-forest flow accumulation exact, without an n*w Phi buffer.
    for (int u = 0; u < n; ++u) {
      int32_t* row = a.flow.data() + (size_t)u*w;
      if (roots[u]) fill(row, row+w, 0);
      else for (int j = 0; j < w; ++j) row[j] = signs[(size_t)u*w+j];
    }
    for (auto it = a.order.rbegin(); it != a.order.rend(); ++it) {
      int u = *it, p = a.next[u]; if (selected[p]) continue;
      int32_t* dst = a.flow.data() + (size_t)p*w; const int32_t* src = a.flow.data() + (size_t)u*w;
      for (int j = 0; j < w; ++j) dst[j] += src[j];
    }
    // E[subtree sources at t] = (W F)_t. This avoids a separate O(w*nnz(F)) multiplication.
    for (int t = 0; t < c; ++t) for (int j = 0; j < w; ++j)
      a.root_sum[(size_t)t*w+j] += a.flow[(size_t)tnodes[t]*w+j];
    for (int u : a.order) {
      int b = bfs.parent[u], src = -1, sign = 0;
      if (a.next[u] == b) { src = u; sign = 1; }
      else if (!roots[b] && a.next[b] == u) { src = b; sign = -1; }
      if (sign) {
        ll* dst = a.edge_sum.data() + (size_t)u*w; const int32_t* f = a.flow.data() + (size_t)src*w;
        for (int j = 0; j < w; ++j) dst[j] += sign * (ll)f[j];
      }
    }
    // Forest Euler intervals for signed path-intersection (Lemma 3.3), NOT tree-path length.
    fill(a.cstart.begin(), a.cstart.end(), 0);
    for (int u : a.order) ++a.cstart[a.next[u]+1];
    for (int u = 0; u < n; ++u) a.cstart[u+1] += a.cstart[u];
    copy(a.cstart.begin(), a.cstart.end()-1, a.position.begin());
    for (int u : a.order) a.children[a.position[a.next[u]]++] = u;
    int clock = 0;
    for (int r = 0; r < n; ++r) if (roots[r]) {
      a.stack.clear(); a.iter.clear(); a.stack.push_back(r); a.iter.push_back(a.cstart[r]); a.tin[r] = clock++;
      while (!a.stack.empty()) {
        int u = a.stack.back();
        if (a.iter.back() < a.cstart[u+1]) {
          int v = a.children[a.iter.back()++]; a.tin[v] = clock++; a.stack.push_back(v); a.iter.push_back(a.cstart[v]);
        } else { a.tout[u] = clock; a.stack.pop_back(); a.iter.pop_back(); }
      }
    }
    auto ancestor = [&](int a0, int u) { return a.tin[a0] <= a.tin[u] && a.tin[u] < a.tout[a0]; };
    for (int u : a.order) {
      int diag = 0;
      for (int x = u; !roots[x]; x = bfs.parent[x]) {
        int b = bfs.parent[x];
        if (a.next[x] == b && ancestor(x, u)) ++diag;
        else if (!roots[b] && a.next[b] == x && ancestor(b, u)) --diag;
      }
      a.z[u] += diag; a.z2[u] += (double)diag*diag;
    }
  }
  void add_forests(ll number) {
    if (number <= 0 || number > UINT32_MAX - forests) throw runtime_error("invalid forest count");
    int nt = (int)min<ll>(threads, number);
    while ((int)workers.size() < nt) workers.emplace_back(new ForestWorker(n, w, c));
#pragma omp parallel num_threads(nt)
    {
      int tid0 = omp_get_thread_num(); auto& a = *workers[tid0]; a.clear_sums();
      // Each worker keeps its own flow/diagonal accumulators. Reduce root labels by vertex
      // after a wave, so the sparse F table is NOT replicated once per worker.
      for (ll wave = 0; wave < number; wave += nt) {
        int live = (int)min<ll>(nt, number-wave);
        if (tid0 < live) sample_one(a, forests + wave + tid0);
#pragma omp barrier
#pragma omp for schedule(static)
        for (int u = 0; u < n; ++u) if (!roots[u]) {
          for (int t = 0; t < live; ++t) {
            int root = workers[t]->root[u];
            if (root >= 0) add_count(counts[u], root, 1);
          }
        }
      }
    }
    for (int j = 0; j < nt; ++j) walk_steps += workers[j]->steps;
#pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) {
      for (int t = 0; t < nt; ++t) {
        const auto& a = *workers[t]; z[u] += a.z[u]; z2[u] += a.z2[u];
        ll* dst = edge_sum.data() + (size_t)u*w; const ll* src = a.edge_sum.data() + (size_t)u*w;
        for (int j = 0; j < w; ++j) dst[j] += src[j];
      }
    }
    for (size_t i = 0; i < root_sum.size(); ++i) for (int t = 0; t < nt; ++t) root_sum[i] += workers[t]->root_sum[i];
    forests += number;
  }
  Estimate estimate(bool retain_y = false) const {
    Estimate out; if (!forests) return out;
    double inv_n = 1.0 / forests;
    out.schur.assign((size_t)c*c, 0);
    vector<double> sink(c, 0);
    // Equation (15), represented as directed effective conductances plus conductance to S.
    for (int i = 0; i < c; ++i) {
      int v = tnodes[i];
      for (ll e = g.ptr[v]; e < g.ptr[v+1]; ++e) {
        int u = g.adj[e];
        if (selected[u]) { sink[i] += 1; continue; }
        if (tid[u] >= 0) { out.schur[(size_t)i*c+tid[u]] -= 1; continue; }
        uint64_t total = 0;
        for (auto [t, count] : counts[u]) { out.schur[(size_t)i*c+t] -= count * inv_n; total += count; }
        sink[i] += (forests - total) * inv_n;
      }
    }
    // Symmetrize edge conductances, retain the estimated grounding conductance, rebuild the diagonal.
    // No diagonal jitter, absolute-value clipping, or unreported exact/iterative-solver fallback.
    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < i; ++j) {
        double x = .5 * (out.schur[(size_t)i*c+j] + out.schur[(size_t)j*c+i]);
        out.schur[(size_t)i*c+j] = out.schur[(size_t)j*c+i] = x;
      }
    }
    for (int i = 0; i < c; ++i) {
      double diag = sink[i];
      for (int j = 0; j < c; ++j) if (i != j) diag -= out.schur[(size_t)i*c+j];
      out.schur[(size_t)i*c+i] = diag;
    }
    // A sampled component can have no observed connection to S. Reject it before
    // roundoff turns a zero Cholesky pivot into a tiny positive, meaningless pivot.
    vector<char> grounded(c, 0); vector<int> queue;
    for (int i = 0; i < c; ++i) if (sink[i] > 0) { grounded[i] = 1; queue.push_back(i); }
    for (size_t h0 = 0; h0 < queue.size(); ++h0) {
      int i = queue[h0];
      for (int j = 0; j < c; ++j) if (!grounded[j] && out.schur[(size_t)i*c+j] < 0) {
        grounded[j] = 1; queue.push_back(j);
      }
    }
    if ((int)queue.size() != c) return out;
    if (!inverse_spd(out.schur, c, out.inverse)) return out;
    vector<double> b((size_t)c*w), h((size_t)c*w, 0);
    for (int t = 0; t < c; ++t) for (int j = 0; j < w; ++j)
      b[(size_t)t*w+j] = source_scale * (root_sum[(size_t)t*w+j]*inv_n + signs[(size_t)tnodes[t]*w+j]);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < c; ++t) {
      double* row = h.data() + (size_t)t*w;
      for (int v = 0; v < c; ++v) {
        double x = out.inverse[(size_t)t*c+v]; const double* src = b.data() + (size_t)v*w;
        for (int j = 0; j < w; ++j) row[j] += x*src[j];
      }
    }
    vector<double> phi((size_t)n*w, 0);
    for (int u : bfs.order) if (!roots[u]) {
      double* row = phi.data() + (size_t)u*w; const double* parent = phi.data() + (size_t)bfs.parent[u]*w;
      const ll* edge = edge_sum.data() + (size_t)u*w;
      for (int j = 0; j < w; ++j) row[j] = parent[j] + edge[j] * (inv_n*source_scale);
    }
    out.diagonal.assign(n, 0); out.score.assign(n, 0);
#pragma omp parallel for schedule(dynamic, 128)
    for (int u = 0; u < n; ++u) {
      if (selected[u]) continue;
      double* y = phi.data() + (size_t)u*w;
      double zu = 0;
      if (tid[u] >= 0) {
        int t = tid[u]; zu = out.inverse[(size_t)t*c+t]; copy(h.begin()+(size_t)t*w, h.begin()+(size_t)(t+1)*w, y);
      } else {
        zu = z[u] * inv_n; const auto& row = counts[u];
        for (size_t i = 0; i < row.size(); ++i) {
          auto [t, count] = row[i]; double f = count*inv_n;
          const double* ht = h.data() + (size_t)t*w;
          for (int j = 0; j < w; ++j) y[j] += f*ht[j];
          double q = f*out.inverse[(size_t)t*c+t];
          for (size_t j = 0; j < i; ++j) q += 2*(row[j].second*inv_n)*out.inverse[(size_t)t*c+row[j].first];
          zu += f*q;
        }
      }
      out.diagonal[u] = zu;
      double norm = 0; for (int j = 0; j < w; ++j) norm += y[j]*y[j];
      if (zu > 0 && isfinite(zu) && isfinite(norm)) out.score[u] = norm / zu;
    }
    double best = -1;
    for (int u = 0; u < n; ++u) {
      out.f_nnz += counts[u].size();
      if (!selected[u] && out.diagonal[u] > 0 && isfinite(out.score[u]) && out.score[u] > best) { out.best=u; best=out.score[u]; }
    }
    out.valid = out.best >= 0;
    if (retain_y) out.y = move(phi);
    return out;
  }
  // Practical selected-denominator proxy, analogous to mode_forest, NOT a confidence interval for Delta.
  // Schur/F contributions are plug-in values; for a T candidate the local forest diagonal variance is zero.
  bool stop_proxy(const Estimate& e, double epsilon) const {
    if (!e.valid) return false;
    int u = e.best; double mean = z[u] / (double)forests;
    double variance = max(0.0, z2[u]/forests - mean*mean), lg = log(3.0*n);
    double error = sqrt(2*variance*lg/forests) + 3.0*stop_depth*lg/forests;
    return error <= epsilon * (e.diagonal[u] - error);
  }
};
#ifndef SCHUR_CFCM_NO_MAIN
struct Args { string graph; int k=20, seed=-1, d=64, c=-1, threads=omp_get_max_threads(); double eps=.2; uint64_t rng=12345; ll maxforests=1<<20; };
static ll parse_integer(const string& s) {
  size_t pos = 0; ll v = stoll(s, &pos); if (pos != s.size()) throw runtime_error("invalid integer: " + s); return v;
}
static Args parse_args(int argc, char** argv) {
  if (argc < 2) throw runtime_error("usage: schur_cfcm <graph.bin> --k 20 --seed V --d 64 --eps .2 [--c auto|C] [--rng 12345] [--maxforests N] [--threads T]");
  Args a; a.graph = argv[1];
  auto small_integer = [&](const string& v) { ll x=parse_integer(v); if(x<INT_MIN || x>INT_MAX) throw runtime_error("integer out of range"); return (int)x; };
  for (int i=2; i<argc; ++i) {
    string key=argv[i]; if (i+1==argc) throw runtime_error("missing value for " + key); string v=argv[++i];
    if(key=="--k") a.k=small_integer(v); else if(key=="--seed") a.seed=small_integer(v); else if(key=="--d") a.d=small_integer(v);
    else if(key=="--c") a.c=v=="auto" ? -1 : small_integer(v); else if(key=="--threads") a.threads=small_integer(v);
    else if(key=="--rng") { size_t p=0; if(v.empty() || v[0]=='-') throw runtime_error("negative RNG seed"); a.rng=stoull(v,&p); if(p!=v.size())throw runtime_error("invalid RNG seed"); }
    else if(key=="--maxforests") a.maxforests=parse_integer(v);
    else if(key=="--eps") {size_t p=0; a.eps=stod(v,&p); if(p!=v.size()) throw runtime_error("invalid eps");}
    else throw runtime_error("unknown option: "+key);
  }
  if(a.seed<0 || a.d<1 || a.k<1 || a.threads<1 || a.c< -1 || !(a.eps>0 && a.eps<1) || a.maxforests<1 || a.maxforests>UINT32_MAX)
    throw runtime_error("invalid parameters (seed required; 0<eps<1; 1<=maxforests<=2^32-1)");
  return a;
}
static void print_ints(const vector<int>& v) { cout << '['; for(size_t i=0;i<v.size();++i) cout << (i?",":"") << v[i]; cout << ']'; }
int main(int argc, char** argv) {
  try {
    Args a=parse_args(argc,argv); omp_set_dynamic(0); omp_set_num_threads(a.threads);
    Graph g=load(a.graph); double start=now_s();
    if(a.seed>=g.n || a.k>g.n || a.c>g.n) throw runtime_error("seed/k/c out of graph range");
    vector<char> selected(g.n,0); selected[a.seed]=1; BFS connected(g,selected);
    RootChoice auxiliary=choose_roots(g,a.c);
    vector<int> set{a.seed}; vector<ll> round_forests, round_steps; vector<double> round_time; vector<string> round_stop;
    ll total_forests=0,total_steps=0; size_t peak_nnz=0;
    for(int round=1;round<a.k;++round) {
      double t0=now_s(); vector<int> t; for(int v:auxiliary.nodes) if(!selected[v]) t.push_back(v);
      SchurSampler sampler(g,selected,t,a.d,a.threads,a.rng,round); Estimate e; string stop;
      ll batch=a.threads;
      for(;;) {
        sampler.add_forests(min(batch,a.maxforests-sampler.forests)); e=sampler.estimate();
        if(e.valid && sampler.stop_proxy(e,a.eps)) {stop="proxy";break;}
        if(sampler.forests==a.maxforests) {
          if(!e.valid) throw runtime_error("no SPD Schur estimate / valid candidate at maxforests; increase cap");
          stop="cap";break;
        }
        batch=min<ll>(batch*2,a.maxforests-sampler.forests);
      }
      selected[e.best]=1;set.push_back(e.best);total_forests+=sampler.forests;total_steps+=sampler.walk_steps;
      round_forests.push_back(sampler.forests);round_steps.push_back(sampler.walk_steps);round_time.push_back(now_s()-t0);round_stop.push_back(stop);peak_nnz=max(peak_nnz,e.f_nnz);
      fprintf(stderr,"schur round %d: v=%d T=%zu forests=%lld walk_steps=%lld stop=%s time=%.3f total=%.3f\n",round,e.best,t.size(),sampler.forests,sampler.walk_steps,stop.c_str(),round_time.back(),now_s()-start);
    }
    cout << setprecision(12) << "{\"mode\":\"schur\",\"n\":" << g.n << ",\"m\":" << g.adj.size()/2 << ",\"k\":" << a.k << ",\"w\":" << a.d << ",\"c\":" << auxiliary.nodes.size()
         << ",\"dmax_T\":" << auxiliary.dmax << ",\"threads\":" << a.threads << ",\"set\":"; print_ints(set);
    cout << ",\"time\":" << now_s()-start << ",\"forests\":" << total_forests << ",\"walk_steps\":" << total_steps << ",\"round_forests\":[";
    for(size_t i=0;i<round_forests.size();++i)cout<<(i?",":"")<<round_forests[i];
    cout<<"],\"round_time\":[";
    for(size_t i=0;i<round_time.size();++i)cout<<(i?",":"")<<round_time[i];
    cout<<"],\"round_walk_steps\":[";
    for(size_t i=0;i<round_steps.size();++i)cout<<(i?",":"")<<round_steps[i];
    cout<<"],\"round_stop\":[";
    for(size_t i=0;i<round_stop.size();++i)cout<<(i?",":"")<<'"'<<round_stop[i]<<'"';
    cout<<"],\"peak_f_nnz\":"<<peak_nnz<<"}\n";
  } catch(const exception& e) {cerr<<"schur_cfcm: "<<e.what()<<'\n';return 1;}
}
#endif

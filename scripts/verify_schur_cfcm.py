#!/usr/bin/env python3
"""Focused SchurCFCM checks, including temporary SNAP downloads and same-machine timing.

Run from workspace root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 uv run -q --no-project --with numpy --with scipy python -I scripts/verify_schur_cfcm.py
All generated graphs, helper binaries and downloads are deleted by TemporaryDirectory.
--quick is a development smoke run, NOT the complete acceptance run.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile
import time
import urllib.request
import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ['g++', '-O3', '-march=native', '-std=c++17', '-fopenmp']


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(cmd, *, timeout=1800, success=True):
    result = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'})
    if success:
        require(result.returncode == 0, f'command failed: {cmd}\n{result.stderr[-5000:]}')
    else:
        require(result.returncode != 0 and not result.stdout.strip(), 'invalid input must fail without JSON')
    return result


def output(cmd, **kwargs):
    result = run(cmd, **kwargs)
    lines = result.stdout.strip().splitlines()
    require(len(lines) == 1, 'stdout must contain exactly one JSON line')
    return json.loads(lines[0])


def graph_file(path, adjacency):
    adjacency = np.asarray(adjacency, dtype=bool)
    require(np.array_equal(adjacency, adjacency.T) and not np.any(np.diag(adjacency)), 'fixture must be simple undirected')
    n = len(adjacency)
    ptr = np.r_[0, np.cumsum(adjacency.sum(axis=1), dtype=np.int64)]
    adj = np.nonzero(adjacency)[1].astype('<i4')
    with open(path, 'wb') as f:
        f.write(struct.pack('<4siq', b'GRND', n, len(adj)))
        f.write(ptr.astype('<i8').tobytes())
        f.write(adj.tobytes())


def fixture(n, seed, kind=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=bool)
    # A spanning cycle makes the graph connected without conditioning the random sample.
    for u in range(n):
        a[u, (u+1) % n] = a[(u+1) % n, u] = True
    probability = np.full((n, n), 6/n)
    if kind == 1:
        labels = np.arange(n) // max(1, n//4)
        probability = np.where(labels[:, None] == labels[None, :], 20/n, 1/n)
    if kind == 2:
        # Hubs and a sparse periphery, rather than only homogeneous random graphs.
        probability[:max(3, n//40), :] = .22
        probability[:, :max(3, n//40)] = .22
    extra = np.triu(rng.random((n, n)) < probability, 1)
    return a | extra | extra.T


def laplacian(a):
    return np.diag(a.sum(axis=1).astype(float)) - a


def grounded_inverse(a, selected):
    alive = np.array([u for u in range(len(a)) if u not in selected])
    r = np.zeros((len(a), len(a)))
    if len(alive):
        matrix = laplacian(a)[np.ix_(alive, alive)]
        r[np.ix_(alive, alive)] = cho_solve(cho_factor(matrix, lower=True), np.eye(len(alive)))
    return r


def exact_greedy(a, k, seed):
    r = grounded_inverse(a, {seed})
    selected = [seed]
    for _ in range(1, k):
        diag = np.diag(r)
        scores = np.divide(np.sum(r*r, axis=0), diag, out=np.full(len(a), -np.inf), where=diag > 1e-14)
        v = int(np.argmax(scores)); selected.append(v)
        x = r[:, v].copy(); r -= np.outer(x, x) / x[v]; r[v, :] = r[:, v] = 0
    return selected


def relative_rmse(x, y):
    return float(np.linalg.norm(np.asarray(x) - y) / np.linalg.norm(y))


def compile_programs(tmp):
    schur, ground, driver = (tmp / name for name in ('schur', 'ground', 'inspect'))
    run(FLAGS + ['-Wall', '-Wextra', '-Werror', '-o', schur, ROOT/'src/schur_cfcm.cpp'])
    run(FLAGS + ['-o', ground, ROOT/'src/ground.cpp'])
    source = tmp/'inspect.cpp'
    source.write_text(r'''
#define SCHUR_CFCM_NO_MAIN
#include "SCHUR_SOURCE"
static vector<int> csv(const string& s) {
  vector<int> out; string v; stringstream f(s); while(getline(f,v,',')) if(!v.empty()) out.push_back(stoi(v)); return out;
}
template<class T> static void array_json(const vector<T>& a) {
  cout << '['; for(size_t i=0;i<a.size();++i)cout<<(i?",":"")<<+a[i]; cout<<']';
}
int main(int argc, char** argv) {
  try {
    if(argc!=9) throw runtime_error("inspect args");
    auto g=load(argv[1]); auto s=csv(argv[2]); vector<char> selected(g.n,0); for(int v:s) selected[v]=1;
    vector<int> t;
    if(string(argv[3])=="auto") {for(int v:choose_roots(g,-1).nodes)if(!selected[v])t.push_back(v);}
    else t=csv(argv[3]);
    int w=stoi(argv[4]), threads=stoi(argv[5]); omp_set_dynamic(0);omp_set_num_threads(threads);
    SchurSampler sampler(g,selected,t,w,threads,stoull(argv[6]),1,stoi(argv[8])!=0);
    for(int target:csv(argv[7])) {
      sampler.add_forests(target-sampler.forests); auto e=sampler.estimate(true);
      if(!e.valid) throw runtime_error("invalid test Schur estimate");
      cout<<setprecision(17)<<"{\"N\":"<<target<<",\"T\":";array_json(t);
      cout<<",\"z\":";array_json(e.diagonal);cout<<",\"score\":";array_json(e.score);
      cout<<",\"y\":";array_json(e.y);cout<<",\"signs\":";array_json(sampler.signs);
      cout<<",\"schur\":";array_json(e.schur);cout<<",\"inverse\":";array_json(e.inverse);
      vector<double> f((size_t)g.n*t.size(),0),base(g.n),wf(t.size()*w);
      for(int u=0;u<g.n;++u){base[u]=sampler.z[u]/(double)target;for(auto [v,count]:sampler.counts[u])f[(size_t)u*t.size()+v]=count/(double)target;}
      for(size_t j=0;j<wf.size();++j)wf[j]=sampler.root_sum[j]*sampler.source_scale/target;
      cout<<",\"F\":";array_json(f);cout<<",\"base\":";array_json(base);cout<<",\"WF\":";array_json(wf);
      cout<<",\"steps\":"<<sampler.walk_steps<<",\"f_nnz\":"<<e.f_nnz<<"}\n";
    }
  }catch(const exception& e){cerr<<e.what()<<'\n';return 1;}
}
'''.replace('SCHUR_SOURCE', str(ROOT/'src/schur_cfcm.cpp')))
    run(FLAGS + ['-o', driver, source])
    return schur, ground, driver


def inspect(driver, graph, s, t, w, threads, seed, checkpoints, identity):
    args = [driver, graph, ','.join(map(str, s)), t if isinstance(t, str) else ','.join(map(str, t)),
            w, threads, seed, ','.join(map(str, checkpoints)), int(identity)]
    return [json.loads(line) for line in run(args).stdout.splitlines()]


def check_interface(tmp, schur, driver):
    a = fixture(32, 42, 2); path = tmp/'interface.bin'; graph_file(path, a)
    required = {'mode', 'n', 'm', 'k', 'w', 'c', 'threads', 'set', 'time', 'forests', 'walk_steps', 'round_forests', 'round_time'}
    for c in ('auto', 0, 31, 32):
        result = output([schur, path, '--k', 10, '--seed', 0, '--d', 64, '--eps', .2, '--c', c, '--threads', 2, '--maxforests', 73])
        require(required <= result.keys(), 'JSON schema')
        require(result['set'][0] == 0 and len(set(result['set'])) == 10, 'cardinality/shared seed')
        require(result['forests'] == sum(result['round_forests']) and len(result['round_time']) == 9, 'round counters')
        require(max(result['round_forests']) <= 73 and min(result['round_forests']) >= 1, 'hard forest cap')
        require(result['walk_steps'] == sum(result['round_walk_steps']), 'walk counters')
    for args in (['--seed', -1], ['--seed', 32], ['--seed', 0, '--d', 0], ['--seed', 0, '--eps', 'nan'],
                 ['--seed', 0, '--k', 33], ['--seed', 0, '--maxforests', 0], ['--seed', 0, '--unknown', 2]):
        run([schur, path, *args], success=False)
    truncated = tmp/'bad.bin'; truncated.write_bytes(path.read_bytes()[:-4]); run([schur,truncated,'--seed',0],success=False)
    disconnected = np.zeros((4,4),bool);disconnected[0,1]=disconnected[1,0]=1;disconnected[2,3]=disconnected[3,2]=1
    graph_file(tmp/'disconnected.bin',disconnected);run([schur,tmp/'disconnected.bin','--seed',0,'--k',2],success=False)
    # Exact degeneracies: U empty (all active vertices in T), T empty, k=1, n=1.
    exact = grounded_inverse(a,{0})
    full = inspect(driver,path,[0],list(range(1,32)),32,2,777,[2],True)[0]
    require(relative_rmse(full['z'], np.diag(exact)) < 2e-12, 'U-empty diagonal')
    require(relative_rmse(np.array(full['y']).reshape(32,32), exact) < 2e-12, 'U-empty inverse')
    require(full['steps']==0,'U-empty sampler')
    one = tmp/'one.bin';graph_file(one,np.zeros((1,1),bool));r=output([schur,one,'--seed',0,'--k',1,'--threads',1])
    require(r['set']==[0] and r['forests']==0,'singleton graph')
    # Same per-forest random streams and integer accumulators at 1 vs 4 threads.
    single=inspect(driver,path,[0,7],'auto',16,1,777,[1024],False)[0]
    parallel=inspect(driver,path,[0,7],'auto',16,4,777,[1024],False)[0]
    for key in ('F','z','score','steps','schur','WF'):
        require(np.allclose(single[key],parallel[key],rtol=2e-13,atol=2e-13),f'thread invariance: {key}')
    # Independent brute-force residual-degree peeling, not a test of a copied C++ rule.
    for seed in range(5):
        a=fixture(50,seed,seed%3);graph_file(path,a);remaining=set(range(50));sequence=[];choices=[]
        for count in range(51):
            degrees={u:int(a[u,list(remaining)].sum()) for u in remaining}
            dm=max(degrees.values(),default=0);choices.append((abs(count-dm),count,dm))
            if remaining:
                u=min(remaining,key=lambda v:(-degrees[v],v));remaining.remove(u);sequence.append(u)
        _,c,dm=min(choices)
        r=output([schur,path,'--seed',0,'--k',1,'--threads',1])
        require((r['c'],r['dmax_T'])==(c,dm),'auto T crossing/degree rule')
        fixed=output([schur,path,'--seed',0,'--k',1,'--threads',1,'--c',7])
        require(fixed['dmax_T']==choices[7][2],'fixed c degree rule')
    # A signed single-forest path estimator need not be nonnegative. This control
    # prevents interpreting the practical Bernstein proxy as a proved CI.
    fixed_path=[(0,1),(1,2),(2,5)]
    forest_path=[(0,3),(3,2),(2,1),(1,4),(4,5)]
    value=sum(int(edge in forest_path)-int(edge[::-1] in forest_path) for edge in fixed_path)
    require(value==-1,'signed path-estimator negative control')
    print('PASS interface, strict cap, edge cases, invalid input, thread invariance, residual-degree T (tol 2e-12)', flush=True)


def check_convergence(tmp, driver, quick):
    specs=[(80,0,True)] if quick else [(200,0,True),(400,1,True),(1000,2,False)]
    checkpoints=[256,1024,4096] if quick else [1024,8192,100000]
    all_errors=[]
    for n,kind,identity in specs:
        a=fixture(n,1900+kind,kind);path=tmp/f'conv-{n}.bin';graph_file(path,a)
        s=[0,n//3,2*n//3];r=grounded_inverse(a,set(s));diag=np.diag(r)
        truth=np.divide(np.sum(r*r,axis=0),diag,out=np.zeros(n),where=diag>0)
        w=n if identity else 64
        errors=[]
        for seed in ([701] if quick else [701,1709]):
            samples=inspect(driver,path,s,'auto',w,4,seed,checkpoints,identity)
            sequence=[]
            for sample in samples:
                t=sample['T'];u=[i for i in range(n) if i not in s and i not in t];c=len(t)
                luu=laplacian(a)[np.ix_(u,u)]
                ru=cho_solve(cho_factor(luu,lower=True),np.eye(len(u)))
                f_exact=np.zeros((n,c));f_exact[u]=ru@a[np.ix_(u,t)]
                f=np.array(sample['F']).reshape(n,c)
                signs=np.array(sample['signs']).reshape(n,w)*(1 if identity else 1/np.sqrt(w))
                y_exact=r@signs
                projected=np.divide(np.sum(y_exact*y_exact,axis=1),diag,out=np.zeros(n),where=diag>0)
                # The identity-source cases compare Delta itself, not merely a fixed JL projection.
                require(not identity or relative_rmse(projected,truth)<1e-12,'identity source marginal')
                wf=np.array(sample['WF']).reshape(c,w)
                require(np.allclose(wf,f.T@signs,rtol=2e-12,atol=2e-12),'sparse F vs root-flow WF identity')
                x=np.array(sample['inverse']).reshape(c,c);schur=np.array(sample['schur']).reshape(c,c)
                require(np.linalg.norm(x@schur-np.eye(c),ord=np.inf)<2e-10,'Cholesky inverse residual')
                require(np.max(np.abs(schur-schur.T))<1e-13,'symmetric Schur')
                require(np.min(schur.sum(axis=1))>-2e-12,'grounding conductance preserved')
                require(np.max(f.sum(axis=1))<=1+1e-14 and np.min(f)>=0,'subprobability root rows')
                base=np.zeros(n);base[u]=np.diag(ru)
                e=[relative_rmse(sample['z'],diag),relative_rmse(sample['score'],projected),relative_rmse(f,f_exact),relative_rmse(sample['base'],base)]
                sequence.append(e)
                print(f'  convergence n={n} rng={seed} N={sample["N"]} identity={identity} rel_RMSE(z,Delta,F,base)='+','.join(f'{v:.5g}' for v in e),flush=True)
            errors.append(sequence)
            final=sequence[-1]
            threshold=.12 if quick else .035
            require(max(final)<threshold,f'final Monte Carlo error n={n}: {final} >= {threshold}')
            # For sign sketches, isolate the nonvanishing JL floor instead of claiming N eliminates it.
            if not identity:
                print(f'  fixed-JL marginal RMSE to full Delta={relative_rmse(projected,truth):.5g}; forest error measured relative to this projection',flush=True)
        aggregate=np.sqrt(np.mean(np.array(errors)**2,axis=0))
        slopes=np.polyfit(np.log(checkpoints),np.log(aggregate),1)[0]
        if not quick:
            require(np.all((slopes < -.25)&(slopes > -.85)),f'expected N^(-1/2) trend: {slopes}')
            require(np.all(aggregate[-1] < aggregate[0]*.3),'error must shrink substantially, not just pass a loose endpoint')
        all_errors.append((n,aggregate[-1],slopes))
        print(f'PASS convergence n={n}, endpoint max={max(aggregate[-1]):.5g}, slopes={slopes.round(3).tolist()}',flush=True)
    return all_errors


def check_quality(tmp, schur, ground, quick):
    quality=[]
    for n,kind in ([(80,2)] if quick else [(200,0),(400,1),(1000,2)]):
        a=fixture(n,730+kind,kind);path=tmp/f'quality-{n}.bin';graph_file(path,a)
        seed=int(np.argmax(a.sum(axis=1)));k=10
        dense=output([ground,'dense',path,'--p',1,'--k',k,'--seed',seed,'--threads',4])
        ref=exact_greedy(a,k,seed)
        tr_ref=float(np.trace(grounded_inverse(a,set(ref))))
        tr_dense=float(np.trace(grounded_inverse(a,set(dense['set']))))
        require(abs(tr_ref/tr_dense-1)<2e-10,'ground dense vs independent rank-one exact greedy')
        for rng in ([12345] if quick else [12345,72931,40961]):
            common=['--seed',seed,'--k',k,'--d',64,'--eps',.2,'--rng',rng,'--threads',4,'--maxforests',8192]
            sr=output([schur,path,*common]);fr=output([ground,'forest',path,*common])
            ts=float(np.trace(grounded_inverse(a,set(sr['set']))));tf=float(np.trace(grounded_inverse(a,set(fr['set']))))
            require(ts<=1.05*tr_dense,f'Schur quality >5% above dense on n={n}, rng={rng}')
            require(tf<=1.05*tr_dense,f'Forest control quality >5% above dense on n={n}, rng={rng}')
            quality.append((n,rng,ts/tr_dense,tf/tr_dense))
            print(f'  quality n={n} rng={rng}: trace exact={tr_dense:.8g} schur={ts:.8g} forest={tf:.8g}; ratios {ts/tr_dense:.6f}/{tf/tr_dense:.6f}',flush=True)
    print('PASS k=10 quality vs independent/ground dense greedy (Schur and Forest trace ratios <=1.05)',flush=True)
    return quality


def download(tmp,name):
    destination=tmp/name
    subdir='bigdata/communities/' if name.startswith('com-') else ''
    url='https://snap.stanford.edu/data/'+subdir+name
    print(f'  downloading {name} from SNAP into temporary workspace directory',flush=True)
    with urllib.request.urlopen(url,timeout=90) as response, open(destination,'wb') as f:
        while chunk:=response.read(1<<20): f.write(chunk)
    return destination


def check_snap(tmp,schur,ground):
    # Table II root counts are a source cross-check, not a time benchmark on these larger graphs.
    for name,n,m,paper_c in [('com-dblp.ungraph.txt.gz',317080,1049866,131),
                            ('com-amazon.ungraph.txt.gz',334863,925872,96),
                            ('as-skitter.txt.gz',1694616,11094209,965)]:
        raw=download(tmp,name);graph=tmp/(name+'.bin')
        cv=output([ground,'convert',raw,graph]);require((cv['n'],cv['m'])==(n,m),'SNAP size after conversion')
        r=output([schur,graph,'--seed',0,'--k',1,'--threads',4])
        require(r['c']==paper_c,f'Table II |T*| mismatch on {name}: got {r["c"]}, expected {paper_c}')
        print(f'PASS Table II {name}: c={r["c"]}, dmax(T)={r["dmax_T"]}',flush=True)
        raw.unlink();graph.unlink()
    raw=download(tmp,'ca-HepPh.txt.gz');graph=tmp/'ca-HepPh.bin'
    cv=output([ground,'convert',raw,graph]);require(cv['n']==11204,'ca-HepPh LCC size')
    # Common seed: same highest-degree vertex for both baselines. Not timed; no per-method choice.
    data=graph.read_bytes();n=struct.unpack_from('<i',data,4)[0];ptr=np.frombuffer(data,dtype='<i8',count=n+1,offset=16)
    seed=int(np.argmax(np.diff(ptr)))
    times={'schur':[],'forest':[]};counts={'schur':[],'forest':[]};steps={'schur':[],'forest':[]}
    # Alternate order to avoid assigning all warm/cache/background effects to one method.
    for index,rng in enumerate((12345,72931,40961)):
        # Forest's existing implementation tests the cap only AFTER a batch. 16380 is
        # exactly 4*(2^12-1), a common batch boundary, so neither method overshoots it.
        common=['--seed',seed,'--k',10,'--d',64,'--eps',.2,'--rng',rng,'--threads',4,'--maxforests',16380]
        for mode in (('schur','forest') if index%2==0 else ('forest','schur')):
            cmd=[schur,graph,*common] if mode=='schur' else [ground,'forest',graph,*common]
            before=time.monotonic();r=output(cmd,timeout=1800);elapsed=time.monotonic()-before
            require(max(r['round_forests'])<=16380,'timing comparison must not benefit from baseline cap overshoot')
            times[mode].append(r['time']);counts[mode].append(r['forests']);steps[mode].append(r['walk_steps'])
            print(f'  ca-HepPh rng={rng} {mode}: time={r["time"]:.4f}s wall={elapsed:.3f}s forests={r["forests"]} steps={r["walk_steps"]} c={r.get("c","-")}',flush=True)
    sm,fm=statistics.median(times['schur']),statistics.median(times['forest'])
    ratio=fm/sm
    print(f'  ca-HepPh n={cv["n"]} m={cv["m"]} seed={seed}: medians Schur={sm:.4f}s Forest={fm:.4f}s speedup={ratio:.4f}x',flush=True)
    for method in times:
        print(f'  {method}: mean={statistics.mean(times[method]):.4f}s range=[{min(times[method]):.4f},{max(times[method]):.4f}]s',flush=True)
    require(ratio>1.0,'Schur must beat Forest in the requested local medium-graph timing test')
    print('PASS local medium-graph speed comparison (3 fixed RNG seeds, median speedup >1; not a universal speed claim)',flush=True)
    return times,counts,steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--section',choices=['all','interface','convergence','quality','snap'],default='all')
    args=parser.parse_args()
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.schur-verify-',dir=ROOT) as directory:
        tmp=Path(directory);schur,ground,driver=compile_programs(tmp)
        if args.section in ('all','interface'): check_interface(tmp,schur,driver)
        if args.section in ('all','convergence'): check_convergence(tmp,driver,args.quick)
        if args.section in ('all','quality'): check_quality(tmp,schur,ground,args.quick)
        if args.section in ('all','snap') and not args.quick: check_snap(tmp,schur,ground)
    scope='SMOKE ONLY' if args.quick else ('FULL' if args.section=='all' else 'PARTIAL '+args.section)
    print(f'PASS SchurCFCM {scope} verification, elapsed={time.monotonic()-started:.2f}s',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        print(f'FAIL SchurCFCM: {error}',flush=True)
        raise

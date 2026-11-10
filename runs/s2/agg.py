import json, math, os, re, glob, statistics as st
D = os.path.dirname(os.path.abspath(__file__))
def J(f):
    try: return json.load(open(f))
    except Exception: return None
def P(f): return [float(l.split()[0]) for l in open(f)] if os.path.exists(f) else None
def pd(a, b):
    d = [y - x for x, y in zip(a, b)]; n = len(d); mu = sum(d) / n
    se = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1) / n); return mu, se
G = ['com-dblp.ungraph', 'com-amazon.ungraph', 'com-youtube.ungraph', 'roadNet-PA']
def row(g, tag):
    j = J(f'{D}/{g}.{tag}.json'); e = J(f'{D}/{g}.{tag}.eval.json')
    return (j or {}).get('time'), (e or {}).get('tr_inv'), (j or {}).get('k'), P(f'{D}/{g}.{tag}.probe')
print('== seeds (jlg - hyb paired, per seed) ==')
for g in G:
    th, tj, ds = [], [], []
    for s in [1001, 1002, 1003, 1004]:
        a, b = row(g, f'hyb_r{s}'), row(g, f'jlg_r{s}')
        if a[0] is None or b[0] is None or a[3] is None or b[3] is None: continue
        mu, se = pd(a[3], b[3]); th.append(a[0]); tj.append(b[0]); ds.append(mu / a[1] * 100)
        print(f'{g:20s} r{s} hyb {a[0]:8.1f}s tr {a[1]:12.1f} | jlg {b[0]:8.1f}s tr {b[1]:12.1f} | diff {mu:10.1f} ± {se:8.1f} z {mu/se:5.1f}')
    if th: print(f'{g:20s} MEAN hyb {st.mean(th):.1f}±{st.pstdev(th):.1f}s jlg {st.mean(tj):.1f}±{st.pstdev(tj):.1f}s speedup {st.mean(tj)/st.mean(th):.2f}x  diff% {st.mean(ds):.3f}±{st.pstdev(ds):.3f}')
print('== K sweep ==')
for g in G:
    for K in [10, 50]:
        a, b = row(g, f'hyb_K{K}'), row(g, f'jlg_K{K}')
        if a[0] is None or b[0] is None: continue
        s = ''
        if a[3] and b[3]: mu, se = pd(a[3], b[3]); s = f'diff {mu:10.1f} ± {se:8.1f} z {mu/se:5.1f}'
        print(f'{g:20s} k={a[2]} hyb {a[0]:8.1f}s tr {a[1]} | jlg {b[0]:8.1f}s tr {b[1]} | {s}')
print('== d sweep (vs paired/hyb d64) ==')
for g in G:
    ref = 'hyb16' if g == 'roadNet-PA' else 'hyb'
    r = P(f'{D}/paired/{g}.{ref}.probe'); rj = J(f'{D}/../s1b/{g}.{ref}.json')
    print(f'{g:20s} d=64 {rj and rj.get("time")}s')
    for d in [16, 32, 128]:
        a = row(g, f'hyb_d{d}')
        if a[0] is None: continue
        s = ''
        if a[3] and r: mu, se = pd(r, a[3]); s = f'diff vs d64 {mu:10.1f} ± {se:8.1f} z {mu/se:5.1f}'
        print(f'{g:20s} d={d:3d} {a[0]:8.1f}s tr {a[1]} | {s}')

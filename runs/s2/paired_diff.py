import math, os, sys
D = os.path.dirname(os.path.abspath(__file__)) + '/paired'
def load(f): return [float(l.split()[0]) for l in open(f)]
pairs = {'com-dblp.ungraph':'hyb','com-amazon.ungraph':'hyb','com-youtube.ungraph':'hyb','web-Google':'hyb','as-skitter':'hyb','roadNet-PA':'hyb16'}
for g, ref in pairs.items():
    fa = f'{D}/{g}.{ref}.probe'
    if not os.path.exists(fa): continue
    a = load(fa)
    for m in ['jlg1','zhou','inc1','incb','skb16','skb64']:
        fb = f'{D}/{g}.{m}.probe'
        if not os.path.exists(fb): continue
        b = load(fb); d = [y - x for x, y in zip(a, b)]; n = len(d)
        mu = sum(d) / n; se = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1) / n)
        print(f"{g:20s} {m:6s}-{ref:5s} = {mu:12.2f}  se {se:9.2f}  z {mu/se:7.1f}")

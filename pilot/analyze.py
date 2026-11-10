import json, glob, sys

for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(f)); m = r["metrics"]; base = m["s1"]
    keys = [k for k in base if k.startswith("heat")] + ["lambda_min", "tr_inv", "tr_pow_-0.25", "tr_pow_-0.5"]
    print(f"\n{f.split('/')[-1][:-5]}  n={r['n']}  (heat/tr: % change vs CFCC set, negative = better; lambda_min: + = better)")
    print("method    " + "".join(f"{k.replace('heat_tau=', 'h@'):>12s}" for k in keys))
    for name in ["degree", "logdet", "s=0.25", "s=0.5", "s=0.75", "s1", "s2"]:
        row = m[name]
        print(f"{name:10s}" + "".join(f"{100 * (row[k] / base[k] - 1):+12.3f}" for k in keys))

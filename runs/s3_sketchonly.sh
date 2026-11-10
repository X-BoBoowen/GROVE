# Table V (update-path ablation): independent sketch-only runs for DBLP and Amazon. NOT YET RUN (server unreachable 2026-11-10).
# Same settings as the main GROVE runs (runs/s1l.sh) with the retained scalar path disabled (--keepexact 0).
export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s3; P=runs/s1b; mkdir -p $O
for g in com-dblp.ungraph com-amazon.ungraph; do
  S0=$(grep -o "\"seed\":[0-9]*" $P/$g.seed.json | cut -d: -f2)
  timeout 60m ./ground_d inc bin/$g.bin --p 1 --k 20 --seed $S0 --d 64 --slack 0 --sketch 1 --keepexact 0 --budget 64 --ac 1 > $O/$g.sk.json 2> $O/$g.sk.err
  grep -o "\"set\":\[[^]]*\]" $O/$g.sk.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.sk.set
  timeout 60m ./ground_d eval bin/$g.bin --set $O/$g.sk.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.sk.eval.json 2>/dev/null
done
echo DONE > $O/done

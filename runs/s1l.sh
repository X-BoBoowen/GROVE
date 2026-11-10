export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b; g=com-youtube.ungraph
S0=$(grep -o "\"seed\":[0-9]*" $O/$g.seed.json | cut -d: -f2)
timeout 60m ./ground_d inc bin/$g.bin --p 1 --k 20 --seed $S0 --d 64 --slack 0 --sketch 1 --budget 64 --ac 1 > $O/$g.hyb.json 2> $O/$g.hyb.err
grep -o "\"set\":\[[^]]*\]" $O/$g.hyb.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.hyb.set
timeout 60m ./ground_d eval bin/$g.bin --set $O/$g.hyb.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.hyb.eval.json 2>/dev/null
echo DONE > $O/done_l

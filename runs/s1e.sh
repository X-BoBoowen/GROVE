export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
until [ -e $O/done_d ]; do sleep 30; done
g=roadNet-PA; SEED=917903
timeout 40m ./ground inc bin/$g.bin --p 1 --k 20 --seed $SEED --d 256 --slack 0 --accept 0.02 --ac 1 --acmin 1 > $O/$g.inc1e.json 2> $O/$g.inc1e.err
grep -o "\"set\":\[[^]]*\]" $O/$g.inc1e.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.inc1e.set
[ -s $O/$g.inc1e.set ] && timeout 30m ./ground eval bin/$g.bin --set $O/$g.inc1e.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.inc1e.eval.json 2>/dev/null
echo DONE > $O/done_e

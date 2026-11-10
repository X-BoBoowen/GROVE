export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
until [ -e $O/done_j ]; do sleep 30; done
g=com-lj.ungraph
timeout 90m ./ground_d inc bin/$g.bin --p 1 --k 20 --seed 21155 --d 64 --slack 0 --sketch 1 --budget 64 --ac 0 > $O/$g.hybj.json 2> $O/$g.hybj.err
grep -o "\"set\":\[[^]]*\]" $O/$g.hybj.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.hybj.set
[ -s $O/$g.hybj.set ] && timeout 60m ./ground_d eval bin/$g.bin --set $O/$g.hybj.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.hybj.eval.json 2>/dev/null
echo DONE > $O/done_k

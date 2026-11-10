export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
until [ -e $O/done_f ]; do sleep 20; done
ev(){ g=$1; m=$2; grep -o "\"set\":\[[^]]*\]" $O/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.$m.set; [ -s $O/$g.$m.set ] && timeout 60m ./ground_b eval bin/$g.bin --set $O/$g.$m.set --probes 128 --rtol 1e-7 --ac 1 > $O/$g.$m.eval.json 2>/dev/null; }
g=com-lj.ungraph; SEED=21155
timeout 90m ./ground_b inc bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --slack 0 --budget 64 --ac 1 > $O/$g.inc1.json 2> $O/$g.inc1.err
timeout 60m ./ground zhou bin/$g.bin --k 20 --seed $SEED --eps 1e-6 > $O/$g.zhou.json 2> $O/$g.zhou.err
ev $g inc1; ev $g zhou
timeout 90m ./ground_b jlg bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --ac 1 > $O/$g.jlg1.json 2> $O/$g.jlg1.err
ev $g jlg1
echo DONE > $O/done_g

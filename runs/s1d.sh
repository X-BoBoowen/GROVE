export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
ev(){ g=$1; m=$2; grep -o "\"set\":\[[^]]*\]" $O/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.$m.set; [ -s $O/$g.$m.set ] && timeout 40m ./ground eval bin/$g.bin --set $O/$g.$m.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.$m.eval.json 2>/dev/null; }
g=roadNet-PA; SEED=917903
timeout 20m ./ground inc bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --slack 0 --ac 1 --acmin 1 > $O/$g.inc1.json 2> $O/$g.inc1.err
g=com-lj.ungraph
./ground seed bin/$g.bin --d 128 --ac 1 --acmin 1 > $O/$g.seed.json 2>&1
SEED=$(grep -o "\"seed\":[0-9]*" $O/$g.seed.json | cut -d: -f2)
timeout 40m ./ground inc bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --slack 0 --ac 1 > $O/$g.inc1.json 2> $O/$g.inc1.err
timeout 40m ./ground zhou bin/$g.bin --k 20 --seed $SEED --eps 1e-6 > $O/$g.zhou.json 2> $O/$g.zhou.err
ev $g inc1; ev $g zhou; ev roadNet-PA inc1
timeout 60m ./ground jlg bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --ac 1 > $O/$g.jlg1.json 2> $O/$g.jlg1.err
ev $g jlg1
echo DONE > $O/done_d

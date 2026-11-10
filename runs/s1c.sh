export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
evalall(){ g=$1; G=bin/$g.bin; for m in inc1 jlg1 inc2 zhou forest; do
    [ -s $O/$g.$m.json ] || continue
    grep -o "\"set\":\[[^]]*\]" $O/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.$m.set
    [ -s $O/$g.$m.set ] && [ ! -s $O/$g.$m.eval.json ] && timeout 2h ./ground eval $G --set $O/$g.$m.set --probes 128 --rtol 1e-7 --ac 1 > $O/$g.$m.eval.json 2>/dev/null
  done; }
evalall com-amazon.ungraph
for g in com-dblp.ungraph com-youtube.ungraph as-skitter web-Google roadNet-PA com-lj.ungraph roadNet-CA; do
  G=bin/$g.bin
  ./ground seed $G --d 128 > $O/$g.seed.json 2>&1
  SEED=$(grep -o "\"seed\":[0-9]*" $O/$g.seed.json | cut -d: -f2)
  timeout 3h ./ground inc $G --p 1 --k 20 --seed $SEED --d 64 --slack 0 --ac 1 > $O/$g.inc1.json 2> $O/$g.inc1.err
  timeout 3h ./ground jlg $G --p 1 --k 20 --seed $SEED --d 64 --ac 1 > $O/$g.jlg1.json 2> $O/$g.jlg1.err
  timeout 3h ./ground inc $G --p 2 --k 20 --seed $SEED --d 128 --slack 0.2 --tol 1e-6 --ac 1 > $O/$g.inc2.json 2> $O/$g.inc2.err
  timeout 2h ./ground zhou $G --k 20 --seed $SEED --eps 1e-6 > $O/$g.zhou.json 2> $O/$g.zhou.err
  evalall $g
done
SEED=$(grep -o "\"seed\":[0-9]*" $O/com-dblp.ungraph.seed.json | cut -d: -f2)
timeout 1h ./ground forest bin/com-dblp.ungraph.bin --k 20 --seed $SEED --d 64 --eps 0.3 --maxforests 16384 > $O/com-dblp.ungraph.forest.json 2> $O/com-dblp.ungraph.forest.err
evalall com-dblp.ungraph
echo ALLDONE > $O/done

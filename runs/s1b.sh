# s=1 (and p=2) head-to-head, k=20, common seed; AC preconditioner for every method that solves systems.
export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
for g in com-amazon.ungraph com-dblp.ungraph com-youtube.ungraph as-skitter web-Google roadNet-PA com-lj.ungraph roadNet-CA; do
  G=bin/$g.bin
  ./ground seed $G --d 128 > $O/$g.seed.json 2>&1
  SEED=$(grep -o "\"seed\":[0-9]*" $O/$g.seed.json | cut -d: -f2)
  timeout 3h ./ground inc $G --p 1 --k 20 --seed $SEED --d 64 --slack 0 --ac 1 > $O/$g.inc1.json 2> $O/$g.inc1.err
  timeout 3h ./ground jlg $G --p 1 --k 20 --seed $SEED --d 64 --ac 1 > $O/$g.jlg1.json 2> $O/$g.jlg1.err
  timeout 3h ./ground inc $G --p 2 --k 20 --seed $SEED --d 128 --slack 0.2 --tol 1e-6 --ac 1 > $O/$g.inc2.json 2> $O/$g.inc2.err
  timeout 2h ./ground zhou $G --k 20 --seed $SEED --eps 1e-6 > $O/$g.zhou.json 2> $O/$g.zhou.err
  TH=14; [ $g = com-lj.ungraph ] && TH=6
  timeout 3h ./ground forest $G --k 20 --seed $SEED --d 64 --eps 0.3 --maxforests 16384 --threads $TH > $O/$g.forest.json 2> $O/$g.forest.err
  for m in inc1 jlg1 inc2 zhou forest; do
    grep -o "\"set\":\[[^]]*\]" $O/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.$m.set
    [ -s $O/$g.$m.set ] && timeout 2h ./ground eval $G --set $O/$g.$m.set --probes 128 --rtol 1e-7 --ac 1 > $O/$g.$m.eval.json 2>/dev/null
  done
done
echo ALLDONE > $O/done

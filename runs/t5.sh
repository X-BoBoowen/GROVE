export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
for g in com-amazon.ungraph com-dblp.ungraph; do
G=bin/$g.bin
SEED=$(grep -o "\"seed\":[0-9]*" runs/s1/com-amazon.ungraph.seed.json | cut -d: -f2)
[ $g = com-dblp.ungraph ] && SEED=$(grep -o "\"seed\":[0-9]*" runs/dblp_seed.json | cut -d: -f2)
./ground inc $G --p 1 --k 20 --seed $SEED --d 64 --slack 0 --ac 1 > runs/$g.inc1_s0.json 2> runs/$g.inc1_s0.err
./ground jlg $G --p 1 --k 20 --seed $SEED --d 64 --ac 1 > runs/$g.jlg1_ac.json 2> runs/$g.jlg1_ac.err
for m in inc1_ac inc1_s0 jlg1_ac; do
  grep -o "\"set\":\[[^]]*\]" runs/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > runs/$g.$m.set
  ./ground eval $G --set runs/$g.$m.set --probes 256 --rtol 1e-8 --ac 1 > runs/$g.$m.eval.json 2>/dev/null
done
done
echo DONE > runs/t5.done

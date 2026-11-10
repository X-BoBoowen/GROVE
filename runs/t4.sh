export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
for g in com-amazon.ungraph com-dblp.ungraph; do
G=bin/$g.bin
SEED=$(grep -o "\"seed\":[0-9]*" runs/s1/com-amazon.ungraph.seed.json | cut -d: -f2)
[ $g = com-dblp.ungraph ] && SEED=$(grep -o "\"seed\":[0-9]*" runs/dblp_seed.json | cut -d: -f2)
./ground inc $G --p 1 --k 20 --seed $SEED --d 64 --slack 0.2 --ac 1 > runs/$g.inc1_ac.json 2> runs/$g.inc1_ac.err
./ground inc $G --p 1 --k 20 --seed $SEED --d 64 --slack 0.2 --accept 0.1 --ac 1 > runs/$g.inc1_ac_acc.json 2> runs/$g.inc1_ac_acc.err
done
echo DONE > runs/t4.done

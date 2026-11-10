export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
G=bin/com-dblp.ungraph.bin
SEED=$(grep -o "\"seed\":[0-9]*" runs/dblp_seed.json | cut -d: -f2)
for p in 1 2 4 8; do
./ground inc $G --p $p --k 20 --seed $SEED --d 128 --slack 0.2 > runs/dblp_inc_p$p.json 2> runs/dblp_inc_p$p.err
done
./ground jlg $G --p 1 --k 20 --seed $SEED --d 64 > runs/dblp_p1_jlg.json 2> runs/dblp_p1_jlg.err
echo ALLDONE > runs/t3.done

export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
for f in data/*.txt.gz; do b=$(basename $f .txt.gz); ./ground convert $f bin/$b.bin; done > runs/convert2.log 2>&1
G=bin/com-dblp.ungraph.bin
./ground seed $G --d 128 > runs/dblp_seed.json
SEED=$(grep -o "\"seed\":[0-9]*" runs/dblp_seed.json | cut -d: -f2)
./ground celf $G --p 1 --k 20 --seed $SEED --d 64 --slack 0.1 > runs/dblp_p1_a.json 2> runs/dblp_p1_a.err
./ground celf $G --p 1 --k 20 --seed $SEED --d 64 --slack 0.3 --rebuild 3 > runs/dblp_p1_b.json 2> runs/dblp_p1_b.err
./ground jlg $G --p 1 --k 20 --seed $SEED --d 64 > runs/dblp_p1_jlg.json 2> runs/dblp_p1_jlg.err
./ground celf $G --p 2 --k 20 --seed $SEED --d 64 --slack 0.1 > runs/dblp_p2_a.json 2> runs/dblp_p2_a.err
./ground jlg $G --p 2 --k 20 --seed $SEED --d 64 > runs/dblp_p2_jlg.json 2> runs/dblp_p2_jlg.err
echo ALLDONE > runs/t2.done

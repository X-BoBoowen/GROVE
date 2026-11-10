export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
./ground celf bin/com-dblp.ungraph.bin --p 1 --k 20 --seed 22804 --d 64 --slack 0.1 > runs/dblp_p1.json 2> runs/dblp_p1.err

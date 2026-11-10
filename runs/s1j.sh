export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
until [ -e $O/done_i ]; do sleep 30; done
ev(){ g=$1; m=$2; grep -o "\"set\":\[[^]]*\]" $O/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.$m.set; [ -s $O/$g.$m.set ] && timeout 60m ./ground_d eval bin/$g.bin --set $O/$g.$m.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.$m.eval.json 2>/dev/null; }
run(){ g=$1; SEED=$2; tag=$3; shift 3; timeout 90m ./ground_d inc bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --slack 0 --sketch 1 --ac 1 "$@" > $O/$g.$tag.json 2> $O/$g.$tag.err; ev $g $tag; }
SK=$(grep -o "\"seed\":[0-9]*" $O/as-skitter.seed.json | cut -d: -f2)
run as-skitter $SK hyb
run roadNet-PA 917903 hyb16 --acmin 1 --budget 16
for g in com-dblp.ungraph com-amazon.ungraph web-Google; do S0=$(grep -o "\"seed\":[0-9]*" $O/$g.seed.json | cut -d: -f2); run $g $S0 hyb --budget 64; done
echo DONE > $O/done_j

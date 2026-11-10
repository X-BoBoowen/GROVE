export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
O=runs/s1b
until [ -e $O/done_g ]; do sleep 30; done
ev(){ g=$1; m=$2; grep -o "\"set\":\[[^]]*\]" $O/$g.$m.json | sed "s/\"set\":\[//;s/\]//;s/,/ /g" > $O/$g.$m.set; [ -s $O/$g.$m.set ] && timeout 60m ./ground_c eval bin/$g.bin --set $O/$g.$m.set --probes 128 --rtol 1e-7 --ac 1 --acmin 1 > $O/$g.$m.eval.json 2>/dev/null; }
run(){ g=$1; SEED=$2; tag=$3; shift 3; timeout 90m ./ground_c inc bin/$g.bin --p 1 --k 20 --seed $SEED --d 64 --slack 0 --sketch 1 --ac 1 "$@" > $O/$g.$tag.json 2> $O/$g.$tag.err; ev $g $tag; }
SK=$(grep -o "\"seed\":[0-9]*" $O/as-skitter.seed.json | cut -d: -f2)
run as-skitter $SK sk
run roadNet-PA 917903 skb64 --acmin 1 --budget 64
run roadNet-PA 917903 skb16 --acmin 1 --budget 16
YT=$(grep -o "\"seed\":[0-9]*" $O/com-youtube.ungraph.seed.json | cut -d: -f2)
run com-youtube.ungraph $YT sk
run com-lj.ungraph 21155 sk --budget 64
echo DONE > $O/done_h

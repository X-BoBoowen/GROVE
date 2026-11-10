export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
B=./ground_f; O=runs/s2; P=runs/s1b
mkdir -p $O/paired
while pgrep -f "[g]round_e forest" >/dev/null; do sleep 10; done
seedof(){ grep -o "\"seed\":[0-9]*" $P/$1.seed.json | cut -d: -f2; }
setof(){ grep -o "\"set\":\[[^]]*\]" $1 | sed "s/\"set\":\[//;s/\]//;s/,/ /g"; }
acx(){ [ "$1" = roadNet-PA ] && echo "--acmin 1 --budget 16" || echo "--budget 64"; }
ev(){ g=$1; f=$2; out=$3; [ -s $f ] && timeout 60m $B eval bin/$g.bin --set $f --probes 128 --rtol 1e-7 --ac 1 --acmin 1 --dump $out.probe > $out.eval.json 2>/dev/null; }
log(){ echo "$(date +%H:%M:%S) $*" >> $O/progress.log; }
# ---- Part 1: paired re-evaluation of the existing main-table sets
log part1 start
for g in com-dblp.ungraph com-amazon.ungraph com-youtube.ungraph web-Google as-skitter; do
  for m in hyb jlg1 zhou inc1; do ev $g $P/$g.$m.set $O/paired/$g.$m; log p1 $g $m; done
done
for m in hyb16 jlg1 zhou incb skb16 skb64; do ev roadNet-PA $P/roadNet-PA.$m.set $O/paired/roadNet-PA.$m; log p1 road $m; done
# ---- Part 2: selection-randomness seeds (4 extra rng values)
runh(){ g=$1; tag=$2; shift 2; S0=$(seedof $g); timeout 120m $B inc bin/$g.bin --p 1 --seed $S0 --slack 0 --sketch 1 --ac 1 $(acx $g) "$@" > $O/$g.$tag.json 2> $O/$g.$tag.err; setof $O/$g.$tag.json > $O/$g.$tag.set; ev $g $O/$g.$tag.set $O/$g.$tag; log $g $tag; }
runj(){ g=$1; tag=$2; shift 2; S0=$(seedof $g); timeout 180m $B jlg bin/$g.bin --p 1 --seed $S0 --ac 1 "$@" > $O/$g.$tag.json 2> $O/$g.$tag.err; setof $O/$g.$tag.json > $O/$g.$tag.set; ev $g $O/$g.$tag.set $O/$g.$tag; log $g $tag; }
log part2 start
for r in 1001 1002 1003 1004; do
  for g in com-dblp.ungraph com-amazon.ungraph com-youtube.ungraph roadNet-PA; do
    runh $g hyb_r$r --k 20 --d 64 --rng $r
    runj $g jlg_r$r --k 20 --d 64 --rng $r
  done
done
# ---- Part 3: total budget K sweep
log part3 start
for K in 10 50; do
  for g in com-dblp.ungraph com-youtube.ungraph roadNet-PA; do
    runh $g hyb_K$K --k $K --d 64
    runj $g jlg_K$K --k $K --d 64
  done
done
# ---- Part 4: sketch dimension sweep (GROVE)
log part4 start
for d in 16 32 128; do
  for g in com-dblp.ungraph com-youtube.ungraph roadNet-PA; do runh $g hyb_d$d --k 20 --d $d; done
done
# ---- Part 5: ForestCFCM (same machine, eps 0.2, w=64)
log part5 start
runf(){ g=$1; lim=$2; S0=$(seedof $g); timeout $lim $B forest bin/$g.bin --k 20 --seed $S0 --d 64 --eps 0.2 > $O/$g.forest.json 2> $O/$g.forest.err; setof $O/$g.forest.json > $O/$g.forest.set; ev $g $O/$g.forest.set $O/$g.forest; log $g forest; }
for g in com-dblp.ungraph com-amazon.ungraph com-youtube.ungraph as-skitter web-Google; do runf $g 120m; done
runf roadNet-PA 60m
log ALLDONE
echo DONE > $O/done_a

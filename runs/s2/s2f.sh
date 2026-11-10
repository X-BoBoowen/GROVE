export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
B=./ground_f; O=runs/s2; P=runs/s1b
log(){ echo "$(date +%H:%M:%S) $*" >> $O/progress.log; }
seedof(){ grep -o "\"seed\":[0-9]*" $P/$1.seed.json | cut -d: -f2; }
setof(){ grep -o "\"set\":\[[^]]*\]" $1 | sed "s/\"set\":\[//;s/\]//;s/,/ /g"; }
ev(){ g=$1; f=$2; out=$3; [ -s $f ] && timeout 60m $B eval bin/$g.bin --set $f --probes 128 --rtol 1e-7 --ac 1 --acmin 1 --dump $out.probe > $out.eval.json 2>/dev/null; }
# take over Part 5 of s2a.sh (adaptive stop never terminates); replace by fixed forest budgets
while ! grep -q "part5 start" $O/progress.log; do sleep 20; done
pkill -f "[s]2a.sh"; sleep 1; pkill -f "[g]round_f forest"; sleep 2
log part5f start
runf(){ g=$1; lim=$2; nf=$3; S0=$(seedof $g); timeout $lim $B forest bin/$g.bin --k 20 --seed $S0 --d 64 --forests $nf > $O/$g.forest$nf.json 2> $O/$g.forest$nf.err; setof $O/$g.forest$nf.json > $O/$g.forest$nf.set; ev $g $O/$g.forest$nf.set $O/$g.forest$nf; log $g forest$nf; }
for g in com-dblp.ungraph com-amazon.ungraph com-youtube.ungraph web-Google as-skitter; do runf $g 120m 1400; done
runf roadNet-PA 60m 1400
runf com-dblp.ungraph 120m 5600
log ALLDONE
echo DONE > $O/done_a

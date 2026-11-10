export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
B=./ground_f; P=runs/s1b; O2=runs/s2
log(){ echo "$(date +%H:%M:%S) $*" >> runs/s2/progress.log; }
seedof(){ grep -o "\"seed\":[0-9]*" $P/$1.seed.json | cut -d: -f2; }
setof(){ grep -o "\"set\":\[[^]]*\]" $1 | sed "s/\"set\":\[//;s/\]//;s/,/ /g"; }
ev(){ g=$1; f=$2; out=$3; [ -s $f ] && timeout 60m $B eval bin/$g.bin --set $f --probes 128 --rtol 1e-7 --ac 1 --acmin 1 --dump $out.probe > $out.eval.json 2>/dev/null; }
while [ ! -f runs/s2/done_a ]; do sleep 30; done
# ---- Part 6: SchurCFCM (Codex reimplementation), strict per-round cap = fixed forest budget
log part6 schur start
runs(){ g=$1; lim=$2; nf=$3; S0=$(seedof $g); timeout $lim ./schur_cfcm bin/$g.bin --k 20 --seed $S0 --d 64 --eps 0.2 --maxforests $nf --threads 14 > $O2/$g.schur$nf.json 2> $O2/$g.schur$nf.err; setof $O2/$g.schur$nf.json > $O2/$g.schur$nf.set; ev $g $O2/$g.schur$nf.set $O2/$g.schur$nf; log $g schur$nf; }
for g in com-dblp.ungraph com-amazon.ungraph com-youtube.ungraph web-Google as-skitter; do runs $g 120m 1400; done
runs roadNet-PA 60m 1400
runs com-dblp.ungraph 120m 5600
runs com-lj.ungraph 120m 1400
# ---- Part 7: old scalar-cache inc on LiveJournal with the 90-minute limit
S0=$(seedof com-lj.ungraph)
timeout 90m $B inc bin/com-lj.ungraph.bin --p 1 --k 20 --seed $S0 --d 64 --slack 0 --ac 1 > $O2/com-lj.ungraph.inc1_90.json 2> $O2/com-lj.ungraph.inc1_90.err
log lj inc1_90
O=runs/s2/med
G="ca-HepPh ca-AstroPh ca-CondMat as-caida20071105 email-Enron p2p-Gnutella31"
log med start
for g in $G; do
  $B seed bin/$g.bin --d 128 --ac 1 --acmin 1 > $O/$g.seed.json 2>&1
  S0=$(grep -o "\"seed\":[0-9]*" $O/$g.seed.json | cut -d: -f2)
  for p in 1 2 3 4 8; do
    timeout 60m $B inc bin/$g.bin --p $p --k 20 --seed $S0 --d 64 --slack 0 --exactinit 1 --ac 1 --acmin 1 > $O/$g.p$p.json 2> $O/$g.p$p.err
    setof $O/$g.p$p.json > $O/$g.p$p.set
  done
  timeout 60m $B zhou bin/$g.bin --k 20 --seed $S0 --eps 1e-6 > $O/$g.zhou.json 2> $O/$g.zhou.err
  setof $O/$g.zhou.json > $O/$g.zhou.set
  for m in p1 p2 p3 p4 p8 zhou; do
    [ -s $O/$g.$m.set ] && timeout 30m $B eval bin/$g.bin --set $O/$g.$m.set --probes 0 --rtol 1e-9 --ac 1 --acmin 1 > $O/$g.$m.eval.json 2>/dev/null
  done
  log med $g
done
log MEDDONE
echo DONE > runs/s2/done_m
log EVERYTHING_DONE
echo DONE > runs/s2/done_all

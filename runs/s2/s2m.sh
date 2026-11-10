export OMP_NUM_THREADS=14
cd /root/autodl-tmp/p9
B=./ground_f; O=runs/s2/med
mkdir -p $O
log(){ echo "$(date +%H:%M:%S) $*" >> runs/s2/progress.log; }
setof(){ grep -o "\"set\":\[[^]]*\]" $1 | sed "s/\"set\":\[//;s/\]//;s/,/ /g"; }
G="ca-HepPh ca-AstroPh ca-CondMat as-caida20071105 email-Enron p2p-Gnutella31"
for g in $G; do [ -s bin/$g.bin ] || $B convert data/$g.txt.gz bin/$g.bin >> $O/convert.log 2>&1; done
while [ ! -f runs/s2/done_a ]; do sleep 30; done
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

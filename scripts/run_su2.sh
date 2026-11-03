#!/usr/bin/env bash
# Run SU2_CFD on N MPI processes and record the peak memory use of the machine.
# Usage (from a case folder):  ../../scripts/run_su2.sh <config.cfg> [nproc]
set -u
CFG=${1:?config file}
NP=${2:-8}
LOG=${CFG%.cfg}.log

base=$(free -m | awk 'NR==2{print $3}')
( peak=0; while sleep 2; do u=$(free -m | awk 'NR==2{print $3}'); [ "$u" -gt "$peak" ] && peak=$u && echo "$peak" > .peak_mem; done ) &
MON=$!

start=$(date +%s)
# --use-hwthread-cpus: count the 16 hardware threads as slots (default is the 8 cores)
mpirun --use-hwthread-cpus -np "$NP" SU2_CFD "$CFG" > "$LOG" 2>&1
status=$?
end=$(date +%s)
kill $MON 2>/dev/null

peak=$(cat .peak_mem 2>/dev/null || echo "$base"); rm -f .peak_mem
echo "exit status        : $status"
echo "wall time          : $((end - start)) s"
echo "memory before run  : ${base} MB"
echo "peak memory in use : ${peak} MB  (SU2 used about $((peak - base)) MB)"
echo "log file           : $LOG"
exit $status

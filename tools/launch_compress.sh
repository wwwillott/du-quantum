#!/bin/bash
# launch GPU compressions in parallel: tools/launch_compress.sh MINUTES file1.json file2.json ...
cd /c/Users/kaisa/Documents/duquantum
MIN=$1; shift
for f in "$@"; do
  b=$(basename $f .json)
  bash tools/gpupy.sh tools/gpu_compress.py $f --minutes $MIN --outdir data/peep_gpu --tag gpu > logs/gpu_compress/run_$b.out 2>&1 &
done
wait

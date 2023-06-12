#!/bin/bash

script=$0
chunk=$1
num_chunks=50000

echo "script=$script"
echo "chunk=$chunk"
echo "num_chunks=$num_chunks"
echo "SGE_TASK_ID=$SGE_TASK_ID"

# This is required to use dotkits inside scripts
source /broad/software/scripts/useuse


  reuse Python-3.9
  reuse Anaconda3

  source activate model

  export THEANO_FLAGS="compiledir=/humgen/diabetes2/users/oliverr/theano/compiledirs/dir.$SGE_TASK_ID"
  python /humgen/diabetes2/users/oliverr/git/phenet/phenet/multi_fit_new.py classify \
    --config-file /humgen/diabetes2/users/oliverr/git/phenet/cfg/lipo_base2_trained.cfg --pymc3 --debug-level 3 \
    --min-traits 3 \
    --num-chunks $num_chunks --chunk "$chunk" \
    --delim ";" --output-file /humgen/diabetes2/users/oliverr/phenet/out/classifieds.local."$chunk"
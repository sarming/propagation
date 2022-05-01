#!/bin/bash -l

source $PWD/config.txt

echo "$TOPIC"
echo "$JOB_ID"
echo "$GRAPH_FILE"
echo "$SOURCEMAP_FILE"
echo "$STATS_FILE"
echo "$SIM_FEATURES"
echo "$SIM_SOURCES"
echo "$SIM_SAMPLES"

export PATH=$HOME/.local/bin:$PATH
# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR
executable=$CURRENT_WORKDIR/src/bin/run

set -x
mpiexec -n $TASKS_PER_NODE python $executable stats \
  $TOPIC --runid $JOB_ID -f $SIM_FEATURES -a $SIM_SOURCES -s $SIM_SAMPLES \
  --graph input/$GRAPH_FILE --source_map input/$SOURCEMAP_FILE --stats  input/$STATS_FILE\
  --outdir output \
  --params output/params-${TOPIC}-${JOB_ID}.csv
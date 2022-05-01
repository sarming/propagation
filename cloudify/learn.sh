#!/bin/bash -l

source $PWD/config.txt

echo "$TOPIC"
echo "$JOB_ID"
echo "$GRAPH_FILE"
echo "$SOURCEMAP_FILE"
echo "$STATS_FILE"
echo "$PARAM_SAMPLES"
echo "$PARAM_EPSILON"

export PATH=$HOME/.local/bin:$PATH
# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR
executable=$CURRENT_WORKDIR/src/bin/run

mpiexec -n $TASKS_PER_NODE $executable learn_discount $TOPIC \
  --runid $JOB_ID -s $PARAM_SAMPLES --epsilon $PARAM_EPSILON \
  --graph input/$GRAPH_FILE --source_map input/$SOURCEMAP_FILE --stats input/$STATS_FILE \
  --outdir output

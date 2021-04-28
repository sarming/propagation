#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate propagation

source $PWD/config.txt

topic="$TOPIC"
id="$JOB_ID"
tasks_per_node="$TASKS_PER_NODE"

graph_file="$(basename -- $GRAPH_URL)" #TODO naming and directory
echo "$graph_file"

if [ -z "$SOURCE_MAP_URL" ] || [ "$SOURCE_MAP_URL" = "default" ]
    then
        source_map_file=""
    else
        source_map_file="$(basename -- $SOURCE_MAP_URL)"
fi
echo "$source_map_file"

if [ -z "$STATS_URL" ] || [ "$STATS_URL" = "default" ]
    then
        stats_file=""
    else
        stats_file="$(basename -- $STATS_URL)"
fi
echo "$stats_file"

features="$SIM_FEATURES"
echo "$features"
sources="$SIM_SOURCES"
echo "$sources"
samples="$SIM_SAMPLES"
echo "$samples"

export PATH=$HOME/.local/bin:$PATH

# DYNAMIC VARIABLES
#cd $CURRENT_WORKDIR
export PYTHONPATH=$CURRENT_WORKDIR"/src"


#srun --mpi=pmix_v3 --nodes=1 --ntasks-per-node=20 python $PYTHONPATH/run.py sim $topic --runid $id -f $features -a $sources -s $samples --graph $CURRENT_WORKDIR/input/$graph_file --source_map $CURRENT_WORKDIR/input/$source_map_file --params $CURRENT_WORKDIR/output/params-$topic-$id.csv --indir $CURRENT_WORKDIR/input --outdir $CURRENT_WORKDIR/output
#TODO source_map_file and params

if [ "$stats_file" = "" ]
    then
        mpirun -n $tasks_per_node python $PYTHONPATH/run.py sim \
        $topic --runid $id -f $features -a $sources -s $samples \
        --graph $CURRENT_WORKDIR/input/$graph_file \
        --indir $CURRENT_WORKDIR/src/data \
        --outdir $CURRENT_WORKDIR/output
    else
        mpirun -n $tasks_per_node python $PYTHONPATH/run.py sim \
        $topic --runid $id -f $features -a $sources -s $samples \
        --graph $CURRENT_WORKDIR/input/$graph_file \
        --stats $CURRENT_WORKDIR/input/$stats_file \
        --indir $CURRENT_WORKDIR/src/data \
        --outdir $CURRENT_WORKDIR/output
fi

#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate propagation

source config.txt

topic="$TOPIC" #"neos_20200311"
id="$JOB_ID" #123 #TODO dynamic numbering
tasks_per_node="$TASKS_PER_NODE"

#graph_file="https://hidalgo1.man.poznan.pl/dataset/02ef431b-7fb5-4fe5-9ea2-828e2038b395/resource/17bf2d21-60ec-42b9-bcfd-c38f478a8485/download/anonymized_outer_graph_neos_20200311.adjlist" 
graph_file="$(basename -- $GRAPH_URL)" #TODO naming and directory
echo "$graph_file"

if [ -z "$SOURCE_MAP_URL" ]
    then
        source_map_file=""
    else
        source_map_file="$(basename -- $SOURCE_MAP_URL)" #""
fi
echo "$source_map_file"
if [ -z "$STATS_URL" ]
    then
        stats_file=""
    else
        stats_file="$(basename -- $STATS_URL)" #""
fi 
echo "$stats_file"

samples="$PARAM_SAMPLES" #20
echo "$samples"
epsilon="$PARAM_EPSILON" #0.1
echo "$epsilon"


export PATH=$HOME/.local/bin:$PATH

# DYNAMIC VARIABLES
#comment out the assignment of the directory
#CURRENT_WORKDIR=$HOME"/SN_cfy_20210409_093752"
#cd $CURRENT_WORKDIR
#CURRENT_WORKDIR=$PWD
current_workdir=$CURRENT_WORKDIR

export PYTHONPATH=$current_workdir"/src"


#srun --mpi=pmix_v3 --nodes=1 --ntasks-per-node=20 python $CURRENT_WORKDIR/src/run.py learn $topic --runid $id -s $samples --epsilon $epsilon --graph $CURRENT_WORKDIR/$graph_file --indir data --outdir output

mpirun -n $tasks_per_node python $PYTHONPATH/run.py learn $topic -s $samples --epsilon $epsilon --graph $current_workdir/input/$graph_file --indir src/data --outdir output


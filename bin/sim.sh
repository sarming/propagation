#!/bin/bash -l

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate propagation

source $PWD/config.txt

topic="$TOPIC"
id="$JOB_ID"

graph_file="$GRAPH_FILE"
echo "$graph_file"
source_map_file="$SOURCEMAP_FILE"
echo "$source_map_file"
stats_file="$STATS_FILE"
echo "$stats_file"
features="$SIM_FEATURES"
echo "$features"
sources="$SIM_SOURCES"
echo "$sources"
samples="$SIM_SAMPLES"
echo "$samples"

export PATH=$HOME/.local/bin:$PATH
# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR
executable=$CURRENT_WORKDIR/src/bin/run

#srun --mpi=pmix_v3 --nodes=1 --ntasks-per-node=20 python $executable sim $topic --runid $id -f $features -a $sources -s $samples --graph input/$graph_file --source_map input/$source_map_file --params output/params-$topic-$id.csv --indir input --outdir output
#TODO source_map_file and params

if [ "$stats_file" == "" ]; then
  mpirun -n $TASKS_PER_NODE python $executable sim \
    $topic --runid $id -f $features -a $sources -s $samples \
    --graph input/$graph_file \
    --indir input \
    --outdir output
else
  mpirun -n $TASKS_PER_NODE python $executable stats \
    $topic --runid $id -f $features -a $sources -s $samples \
    --graph input/$graph_file \
    --stats input/$stats_file \
    --indir input \
    --outdir output \
    --params output/params-${topic}-${id}.csv
fi

#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate propagation

source config.txt

topic="$TOPIC"
id="$JOB_ID" #TODO dynamic numbering
graph_file="$(basename -- $GRAPH_URL)" #TODO naming and directory

if [ -z "$SOURCE_MAP_URL" ] || [ "$SOURCE_MAP_URL" = "default" ]
    then
        source_map_file=""
    else
        source_map_file="$(basename -- $SOURCE_MAP_URL)" 
fi

features="$SIM_FEATURES"
sources="$SIM_SOURCES"
samples="$SIM_SAMPLES"


export PATH=$HOME/.local/bin:$PATH

# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR

export PYTHONPATH=$CURRENT_WORKDIR"/src"


#srun python run.py sim $topic -id $id -f $features -a $sources -s $samples --graph $graph_file --source_map $source_map_file --params output/params-$topic-$id.csv --indir input --outdir output
#TODO source_map_file and params

mpirun -n $SLURM_NTASKS python $PYTHONPATH/run.py sim $topic -f $features -a $sources -s $samples --graph $CURRENT_WORKDIR/input/$graph_file --indir src/data --outdir output


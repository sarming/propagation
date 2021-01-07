#!/usr/bin/env bash
topic=$1
id=$2
graph_file=`basename $3`
source_map_file=`basename $4`
stats_file=`basename $5`
samples=$6
epsilon=$7

export PATH=$HOME/.local/bin:$PATH

# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR

export PYTHONPATH=$PWD/src:$PYTHONPATH

srun python run.py learn $topic -id $id -s $samples --epsilon $epsilon --graph $graph_file --source_map $source_map_file --stats $stats_file --indir input --outdir output

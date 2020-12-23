#!/usr/bin/env bash
topic=$1
id=$2
graph_file=`basename $3`
source_map_file=`basename $4`
features=$5
sources=$6
samples=$7

export PATH=$HOME/.local/bin:$PATH

# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR

export PYTHONPATH=$PWD/src:$PYTHONPATH

srun python run.py sim $topic -id $id -f $features -a $sources -s $samples --graph $graph_file --source_map $source_map_file --params output/params-$topic-$id.csv --indir input --outdir output

#!/bin/bash

#module load openmpi/4.0.0_gcc620
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate propagation

topic="$1" #"neos_20200311"
echo "$topic"
id="$2" #123 #TODO dynamic numbering
echo "$id"
#graph_file="https://hidalgo1.man.poznan.pl/dataset/02ef431b-7fb5-4fe5-9ea2-828e2038b395/resource/17bf2d21-60ec-42b9-bcfd-c38f478a8485/download/anonymized_outer_graph_neos_20200311.adjlist" 
graph_file="$(basename -- $3)" #TODO naming and directory
echo "$graph_file"
source_map_file="$(basename -- $4)" #""
echo "$source_map_file"
stats_file="$(basename -- $5)" #"" 
echo "$stats_file"
samples="$6" #20
echo "$samples"
epsilon="$7" #0.1
echo "$epsilon"


export PATH=$HOME/.local/bin:$PATH

# DYNAMIC VARIABLES
#comment out the assignment of the directory
#CURRENT_WORKDIR=$HOME"/SN_cfy_20210121_093059"
cd $CURRENT_WORKDIR

export PYTHONPATH=$CURRENT_WORKDIR"/src"


#srun --mpi=pmix_v3 --nodes=1 --ntasks-per-node=20 python $CURRENT_WORKDIR/src/run.py learn $topic --runid $id -s $samples --epsilon $epsilon --graph $CURRENT_WORKDIR/$graph_file --indir data --outdir output

mpirun -n $SLURM_NTASKS python $PYTHONPATH/run.py learn $topic -s $samples --epsilon $epsilon --graph $CURRENT_WORKDIR/input/$graph_file --indir src/data --outdir output


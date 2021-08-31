#!/bin/bash -l

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate propagation

source $PWD/config.txt

topic="$TOPIC" #"neos"
id="$JOB_ID"   #123

graph_file="$GRAPH_FILE"
echo "$graph_file"
samples="$PARAM_SAMPLES" #20
echo "$samples"
epsilon="$PARAM_EPSILON" #0.1
echo "$epsilon"

export PATH=$HOME/.local/bin:$PATH
# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR
executable=$CURRENT_WORKDIR/bin/run

#srun --mpi=pmix_v3 --nodes=1 --ntasks-per-node=20 $executable learn_discount $topic --runid $id -s $samples --epsilon $epsilon --graph input/$graph_file --indir input --outdir output

mpirun -n $TASKS_PER_NODE $executable learn_discount $topic \
  --runid $id -s $samples --epsilon $epsilon \
  --graph input/$graph_file \
  --indir input \
  --outdir output

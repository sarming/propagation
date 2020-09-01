#!/bin/bash
#SBATCH --partition=hidalgo
#SBATCH --time=01:30:00
#SBATCH --nodes=4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=propagation

worker_num=3 # Must be one less that the total number of nodes


 . /work/miniconda3/etc/profile.d/conda.sh
conda activate propagation

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
RAY_ADDRESS=$ip_prefix$suffix

export RAY_ADDRESS

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 & # Starting the head
sleep 10
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$RAY_ADDRESS & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
done
sleep 5
python simulation.py

#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sarming@cs.uni-salzburg.at
#SBATCH --no-requeue

#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env

#SBATCH --account=pn29wi

#SBATCH --time={walltime}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={procs}
#SBATCH --job-name={jobname}
#SBATCH -e slurm/%x_%j.err
#SBATCH -o slurm/%x_%j.out
 
#Important
module load slurm_setup
 
source ../conda/propagation/bin/activate

export RUNID={jobname}_$SLURM_JOB_ID
mkdir out/$RUNID
OUTFILE=out/$RUNID/out-$RUNID-$(date +"%Y%m%dT%H%M").txt

echo nodes: {nodes} > $OUTFILE
echo mpiprocs: "$SLURM_NPROCS" >> $OUTFILE
echo walltime: {walltime} >>$OUTFILE
echo jobid: "$SLURM_JOB_ID" >>$OUTFILE

mpiexec {mpiargs} bin/run {args} --outdir=out/$RUNID &>> $OUTFILE
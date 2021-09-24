#!/bin/bash
#SBATCH --time={walltime}
#SBATCH --nodes={nodes}
#SBATCH --ntask-per-node={procs}
#SBATCH --job-name={jobname}
#SBATCH -e slurm/%x_%j.err
#SBATCH -o slurm/%x_%j.out


 . /work/miniconda3/etc/profile.d/conda.sh
conda activate prop_mpi

export RUNID={jobname}_$SLURM_JOB_ID
mkdir out/$RUNID
OUTFILE=out/$RUNID/out-$RUNID-$(date +"%Y%m%dT%H%M").txt

echo nodes: {nodes} > $OUTFILE
echo mpiprocs: "$SLURM_NPROCS" >> $OUTFILE
echo walltime: {walltime} >>$OUTFILE
echo jobid: "$SLURM_JOB_ID" >>$OUTFILE
echo date: $(date +"%Y-%m-%dT%H:%M:%S") >>$OUTFILE

mpiexec {mpiargs} bin/run {args} --outdir=out/$RUNID &>> $OUTFILE
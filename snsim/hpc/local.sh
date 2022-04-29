#!/bin/bash

DATE=`date +%s`
echo $DATE
export RUNID={jobname}_$DATE
mkdir out/$RUNID
OUTFILE=out/$RUNID/out-$RUNID-$(date +"%Y%m%dT%H%M").txt

echo nodes: {nodes} > $OUTFILE
echo mpiprocs: {nodes} >> $OUTFILE
echo walltime: {walltime} >>$OUTFILE
echo jobid: "$DATE" >>$OUTFILE

mpiexec {mpiargs} -n {nodes} {exe} {args} --outdir=out/$RUNID >> $OUTFILE 2>&1
#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --array 1-1
#SBATCH -t 00:30:00
#SBATCH -J izhi
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%A_%a.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%A_%a.err"

set -e

cd /global/cscratch1/sd/vbaratha/izhi

RUNDIR=runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir $RUNDIR

M_TYPE="L5_TTPC1"
E_TYPE="cADpyr"
# M_TYPE="L1_HAC"
# E_TYPE=cIR
DSET_NAME=${M_TYPE}_${E_TYPE}
NSAMPLES=5
stimname=chaotic_1
stimfile=stims/${stimname}.csv

METADATA_FILE = $RUNDIR/meta.yaml
# OUTFILE=$RUNDIR/${DSET_NAME}_${stimname}.h5
FILENAME=${DSET_NAME}_${stimname}
OUTFILE=$RUNDIR/$FILENAME_\{NODEID\}.h5
echo "STIM FILE" $stimfile
echo "OUTFILE" $OUTFILE
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
args="--outfile $OUTFILE --stim-file ${stimfile} --stim-multiplier 2.0 \
      --model BBP --m-type ${M_TYPE} --e-type ${E_TYPE} --cell-i 2 \
      --num $NSAMPLES --trivial-parallel --print-every 10 \
      --metadata-file ${METADATA_FILE}"

# srun --label -n 64 --ntasks-per-node 1 python run.py $args --create # create output file
srun --label -n 4096 --ntasks-per-node 64 python run.py $args


echo "rawPath: $RUNDIR" >> $METADATA_FILE
echo "rawDataName: ${FILENAME}_*" >> $METADATA_FILE
echo "stimName: $stimname" >> $METADATA_FILE


chmod -R a+r $RUNDIR

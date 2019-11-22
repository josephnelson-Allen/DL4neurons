#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 64
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
NSAMPLES=41
stimname=chirp23a
stimfile=stims/${stimname}.csv

# OUTFILE=$RUNDIR/${DSET_NAME}_${stimname}.h5
OUTFILE=$RUNDIR/${DSET_NAME}_${stimname}_\{NODEID\}.h5
echo "STIM FILE" $stimfile
echo "OUTFILE" $OUTFILE
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
args="--outfile $OUTFILE --stim-file ${stimfile} --stim-multiplier 2.0 \
      --model BBP --m-type ${M_TYPE} --e-type ${E_TYPE} --cell-i 2 \
      --num $NSAMPLES --trivial-parallel --print-every 10"

# srun --label -n 64 --ntasks-per-node 1 python run.py $args --create # create output file
srun --label -n 4096 --ntasks-per-node 64 python run.py $args 

chmod -R a+r $RUNDIR

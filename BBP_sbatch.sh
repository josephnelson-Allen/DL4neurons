#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 100
#SBATCH --array 4-8
#SBATCH -t 04:00:00
#SBATCH -J izhi
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%A_%a.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%A_%a.err"

set -e

# All paths relative to this, prepend this for full path name
IZHI_WORKING_DIR=/global/cscratch1/sd/vbaratha/izhi
cd $IZHI_WORKING_DIR

RUNDIR=runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $RUNDIR

# M_TYPE="L5_TTPC1"
# E_TYPE="cADpyr"
# M_TYPE="L1_HAC"
# E_TYPE=cIR
M_TYPE=$(python cori_get_cell.py --m-type)
E_TYPE=$(python cori_get_cell.py --e-type)
DSET_NAME=${M_TYPE}_${E_TYPE}
NSAMPLES=100
stimname=chaotic_1
stimfile=stims/${stimname}.csv

echo
env | grep SLURM
echo

# OUTFILE=$RUNDIR/${DSET_NAME}_${stimname}.h5
FILENAME=${DSET_NAME}_${stimname}
METADATA_FILE=$RUNDIR/${FILENAME}_meta.yaml
OUTFILE=$RUNDIR/${FILENAME}_\{NODEID\}.h5
echo "M-TYPE" ${M_TYPE}
echo "E-TYPE" ${E_TYPE}
echo "STIM FILE" $stimfile
echo "OUTFILE" $OUTFILE
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
args="--outfile $OUTFILE --stim-file ${stimfile} \
      --model BBP --m-type ${M_TYPE} --e-type ${E_TYPE} --cell-i 0 \
      --num $NSAMPLES --trivial-parallel --print-every 5 \
      --metadata-file ${METADATA_FILE}"

# srun --label -n 64 --ntasks-per-node 1 python run.py $args --create # create output file
srun -n $((${SLURM_NNODES}*64)) --ntasks-per-node 64 python run.py $args
wait


echo "rawPath: ${IZHI_WORKING_DIR}/$RUNDIR" >> $METADATA_FILE
echo "rawDataName: ${FILENAME}_*" >> $METADATA_FILE
echo "stimName: $stimname" >> $METADATA_FILE


chmod a+rx $RUNDIR
chmod -R a+r $RUNDIR/*

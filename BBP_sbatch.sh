#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 4
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

# All paths relative to this, prepend this for full path name
IZHI_WORKING_DIR=/global/cscratch1/sd/vbaratha/izhi
cd $IZHI_WORKING_DIR

RUNDIR=runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $RUNDIR

M_TYPE=$(python cori_get_cell_full.py --m-type)
E_TYPE=$(python cori_get_cell_full.py --e-type)
BBP_NAME=$(python cori_get_cell_full.py --bbp-name)
DSET_NAME=${M_TYPE}_${E_TYPE}
NSAMPLES=1
stimname=chaotic_1
stimfile=stims/${stimname}.csv

echo
env | grep SLURM
echo

export CELLS_PER_JOB=5

for i in $(seq 1 ${CELLS_PER_JOB});
do 
    RUNDIR=runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$i
    mkdir $RUNDIR

    M_TYPE=$(python cori_get_cell_full.py $i --m-type)
    E_TYPE=$(python cori_get_cell_full.py $i --e-type)
    BBP_NAME=$(python cori_get_cell_full.py $i --bbp-name)
    DSET_NAME=${M_TYPE}_${E_TYPE}
    NSAMPLES=100
    NRUNS=1
    NSAMPLES_PER_RUN=$(($NSAMPLES/$NRUNS))
    stimname=chaotic_1
    stimfile=stims/${stimname}.csv

    echo
    env | grep SLURM
    echo

    FILENAME=${BBP_NAME}_${stimname}
    METADATA_FILE=$RUNDIR/${FILENAME}_meta.yaml
    OUTFILE=$RUNDIR/${FILENAME}_\{NODEID\}.h5
    echo "M-TYPE" ${M_TYPE}
    echo "E-TYPE" ${E_TYPE}
    echo "BBP NAME" ${BBP_NAME}
    echo "STIM FILE" $stimfile
    echo "OUTFILE" $OUTFILE
    echo "SLURM_NODEID" ${SLURM_NODEID}
    echo "SLURM_PROCID" ${SLURM_PROCID}
    args="--outfile $OUTFILE --stim-file ${stimfile} \
      --model BBP --m-type ${M_TYPE} --e-type ${E_TYPE} --cell-i 0 \
      --num ${NSAMPLES_PER_RUN} --trivial-parallel --print-every 25 \
      --metadata-file ${METADATA_FILE}"

    for j in $(seq 1 ${NRUNS});
    do
	srun -n $((${SLURM_NNODES}*64)) --ntasks-per-node 64 python run.py $args
    done

    echo "rawPath: ${IZHI_WORKING_DIR}/$RUNDIR" >> $METADATA_FILE
    echo "rawDataName: ${FILENAME}_*" >> $METADATA_FILE
    echo "stimName: $stimname" >> $METADATA_FILE

    chmod a+rx $RUNDIR
    chmod -R a+r $RUNDIR/*
done

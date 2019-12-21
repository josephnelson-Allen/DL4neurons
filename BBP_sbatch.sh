#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH -J DL4N_9cells
#SBATCH -L SCRATCH,project
#SBATCH -C knl
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/DL4neurons/runs/slurm/%A.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/DL4neurons/runs/slurm/%A.err"

set -e

# Stuff for knl
export OMP_NUM_THREADS=1
module unload craype-hugepages2M

# All paths relative to this, prepend this for full path name
IZHI_WORKING_DIR=/global/cscratch1/sd/vbaratha/DL4neurons
cd $IZHI_WORKING_DIR

CELLS_FILE='9cells.txt'

i=0
while read line;
do
    i=$((i+1))

    echo "RUNNING CELL $i OF $(wc -l < ${CELLS_FILE})"

    # M_TYPE=$(python cori_get_cell_full.py $i --m-type)
    # E_TYPE=$(python cori_get_cell_full.py $i --e-type)
    # BBP_NAME=$(python cori_get_cell_full.py $i --bbp-name) 
    BBP_NAME=$(echo $line | awk -F "," '{print $1}')
    M_TYPE=$(echo $line | awk -F "," '{print $2}')
    E_TYPE=$(echo $line | awk -F "," '{print $3}')
    # NSAMPLES=$(echo $line | awk -F "," '{print $4}')

    TOP_RUNDIR=runs/${SLURM_JOBID}
    RUNDIR=$TOP_RUNDIR/${BBP_NAME}
    mkdir -p $RUNDIR
    
    echo $BBP_NAME $M_TYPE $E_TYPE

    DSET_NAME=${M_TYPE}_${E_TYPE}
    NSAMPLES=8
    NRUNS=1
    NSAMPLES_PER_RUN=$(($NSAMPLES/$NRUNS))
    stimname=chaotic_1
    stimfile=stims/${stimname}.csv

    THREADS_PER_NODE=128

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
	srun -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) \
	     --ntasks-per-node ${THREADS_PER_NODE} \
	     python run.py $args
    done

    echo "rawPath: ${IZHI_WORKING_DIR}/$RUNDIR" >> $METADATA_FILE
    echo "rawDataName: ${FILENAME}_*" >> $METADATA_FILE
    echo "stimName: $stimname" >> $METADATA_FILE

    chmod a+rx ${TOP_RUNDIR}
    chmod a+rx $RUNDIR
    chmod -R a+r $RUNDIR/*
done < ${CELLS_FILE}

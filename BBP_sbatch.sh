#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 10
#SBATCH -t 01:00:00
#SBATCH -J DL4N_shifter_test
#SBATCH -L SCRATCH,project
#SBATCH -C knl
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/DL4neurons/runs/slurm/%A.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/DL4neurons/runs/slurm/%A.err"
#SBATCH --image=balewski/ubu18-py3-mpich:v2


# Stuff for knl
export OMP_NUM_THREADS=1
module unload craype-hugepages2M

# All paths relative to this, prepend this for full path name
WORKING_DIR=/global/cscratch1/sd/vbaratha/DL4neurons
cd $WORKING_DIR

CELLS_FILE='allcells.csv'
START_CELL=85
NCELLS=48
END_CELL=$((${START_CELL}+${NCELLS}))
NSAMPLES=16
NRUNS=2
NSAMPLES_PER_RUN=$(($NSAMPLES/$NRUNS))

export THREADS_PER_NODE=128

# prep for shifter
if [ -f ./shifter_env.sh ]; then
    source ./shifter_env.sh
    PYTHON="shifter python3"
else
    PYTHON=python
fi
mkdir -p $RUNDIR

stimname=chaotic_2
stimfile=stims/${stimname}.csv

echo
env | grep SLURM
echo

echo "Using" $PYTHON

FILENAME=\{BBP_NAME\}/\{BBP_NAME\}-${stimname}
METADATA_FILE=$RUNDIR/${FILENAME}-meta.yaml
OUTFILE=$RUNDIR/${FILENAME}-\{NODEID\}.h5
echo "STIM FILE" $stimfile
echo "OUTFILE" $OUTFILE
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
args="--outfile $OUTFILE --stim-file ${stimfile} --model BBP \ 
      --cori-csv ${CELLS_FILE} --cori-start ${START_CELL} --cori-end ${END_CELL} \
      --num ${NSAMPLES_PER_RUN} --trivial-parallel --print-every 2 \
      --metadata-file ${METADATA_FILE}"

for j in $(seq 1 ${NRUNS});
do
    srun --input none -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) \
	 --ntasks-per-node ${THREADS_PER_NODE} \
	 $PYTHON run.py $args
done

echo "rawPath: ${WORKING_DIR}/$RUNDIR" >> $METADATA_FILE
echo "rawDataName: ${FILENAME}_*" >> $METADATA_FILE
echo "stimName: $stimname" >> $METADATA_FILE

chmod a+rx $RUNDIR
chmod a+rx $RUNDIR/*
chmod a+r $RUNDIR/*/*.h5
chmod a+r $RUNDIR/*/*.yaml


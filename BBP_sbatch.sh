#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 2000
#SBATCH -t 10:00:00
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
NSAMPLES=120
NRUNS=3
NSAMPLES_PER_RUN=$(($NSAMPLES/$NRUNS))

echo "CELLS_FILE" ${CELLS_FILE}
echo "START_CELL" ${START_CELL}
echo "NCELLS" ${NCELLS}
echo "END_CELL" ${END_CELL}

export THREADS_PER_NODE=128

# prep for shifter
if [ -f ./shifter_env.sh ]; then
    source ./shifter_env.sh
    PYTHON="shifter python3"
else
    PYTHON=python
fi

# Create all outdirs
echo "Making outdirs at" `date`
RUNDIR=runs/${SLURM_JOBID}
mkdir -p $RUNDIR
for i in $(seq ${START_CELL} ${END_CELL});
do
    i=$(( $i+1 ))
    line=$(head -$i ${CELLS_FILE} | tail -1)
    bbp_name=$(echo $line | awk -F "," '{print $1}')
    mkdir -p $RUNDIR/$bbp_name
done
echo "Done making outdirs at" `date`

stimname=chaotic_2
stimfile=stims/${stimname}.csv

echo
env | grep SLURM
echo

echo "Using" $PYTHON

FILENAME=\{BBP_NAME\}-${stimname}
METADATA_FILE=$RUNDIR/${FILENAME}-meta.yaml
OUTFILE=$RUNDIR/\{BBP_NAME\}/${FILENAME}-\{NODEID\}.h5
echo "STIM FILE" $stimfile
echo "OUTFILE" $OUTFILE
echo "SLURM_NODEID" ${SLURM_NODEID}
echo "SLURM_PROCID" ${SLURM_PROCID}
args="--outfile $OUTFILE --stim-file ${stimfile} --model BBP \
      --cori-csv ${CELLS_FILE} --cori-start ${START_CELL} --cori-end ${END_CELL} \
      --num ${NSAMPLES_PER_RUN} --trivial-parallel --print-every 2 \
      --metadata-file ${METADATA_FILE}"
echo "args" $args

for j in $(seq 1 ${NRUNS});
do
    echo "Doing run $j of $NRUNS at" `date`
    srun --input none -k -n $((${SLURM_NNODES}*${THREADS_PER_NODE})) \
	 --ntasks-per-node ${THREADS_PER_NODE} \
	 $PYTHON run.py $args
    echo "Done run $j of $NRUNS at" `date`

done

chmod a+rx $RUNDIR
chmod a+rx $RUNDIR/*
chmod a+r $RUNDIR/*/*.h5
chmod a+r $RUNDIR/*/*.yaml


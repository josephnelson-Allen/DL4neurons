#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J izhi
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%j.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%j.err"

cd /global/cscratch1/sd/vbaratha/izhi

MODELNAME=hh_point_5param
VERSION=2

RUNDIR=runs/${SLURM_JOB_ID}
mkdir $RUNDIR
DSET_NAME=${MODELNAME}_v$VERSION
NSAMPLES=10000

stimname=chirp_damp_10k
stimfile=stims/${stimname}.csv
paramfile=$RUNDIR/${DSET_NAME}.csv
OUTFILE=$RUNDIR/${DSET_NAME}_${stimname}.h5
echo "STIM FILE" $stimfile
echo "OUTFILE" $OUTFILE
args="--outfile $OUTFILE --stim-file ${stimfile} --param-file ${paramfile} \
      --model $MODELNAME --print-every 10"

srun -n 1 python runpy $args --create-params
srun -n 1 python run.py $args --create
srun --label -n 64 python run.py $args 

chmod -R a+r $RUNDIR

# for stim in $(ls stims)
# do
#     stimname=`echo $stim | sed s/.csv//`
#     OUTFILE=$RUNDIR/${DSET_NAME}_${stimname}.h5
    
#     ## Report the stim name
#     echo "STIM FILE" $stim
#     echo "OUTFILE" $OUTFILE

#     args="--outfile $OUTFILE --stim-file stims/$stim --param-file params/izhi_v3.csv"

#     ## Create the output file
#     srun -n 1 python run.py $args --create

#     ## Run the simulation
#     srun --label -n 128 python run.py $args

# done


# for stim_type in "ramp" "step" "noise"
# do
#     for i in {00..07}
#     do
#         OUTFILE=$RUNDIR/izhi_${stim_type}_${i}.h5
        
#         ## Report the stim type/idx
#         echo "STIM" $stim_type $i
#         echo "OUTFILE" $OUTFILE

#         args="--outfile $OUTFILE --stim-type $stim_type --stim-idx $i"
        
#         ## Create the output file
#         srun -n 1 python run.py $args --create

#         ## Run the simulation
#         srun --label -n 64 python run.py $args --param-sweep
#     done
# done

# stim_type='step'
# i=7
# srun --label -n 64 python run.py \
#      --outfile $OUTFILE --stim-type $stim_type --stim-idx $i --param-sweep


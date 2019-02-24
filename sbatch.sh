#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH -J izhi
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --oversubscribe
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%j.out"
#SBATCH --error "/global/cscratch1/sd/vbaratha/izhi/runs/slurm/%j.err"

cd /global/cscratch1/sd/vbaratha/izhi

mkdir runs/${SLURM_JOB_ID}
OUTFILE=runs/${SLURM_JOB_ID}/data.nwb

declare -a arr=("ramp" "step" "noise")

## Create the output file
python run.py --outfile $OUTFILE --create

for stim_type in "${arr[@]}"
do
    for i in seq 0 7
    do
        srun --oversubscribe --label -n 10000 python param_sweep.py \
             --outfile $OUTFILE --stim-type $stim_type --stim-idx $i
    done
done



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

mkdir runs/${SLURM_JOB_ID}
OUTFILE=runs/${SLURM_JOB_ID}/izhi_sim_data.nwb

## Create the output file
srun -n 1 python run.py --outfile $OUTFILE --create

declare -a arr=("ramp" "step" "noise")

# for stim_type in "${arr[@]}"
# do
#     for i in seq 0 7
#     do
#         srun --label -n 128 python param_sweep.py \
#              --outfile $OUTFILE --stim-type $stim_type --stim-idx $i --param-sweep
#     done
# done

stim_type='step'
i=7
srun --label -n 64 python run.py \
     --outfile $OUTFILE --stim-type $stim_type --stim-idx $i --param-sweep


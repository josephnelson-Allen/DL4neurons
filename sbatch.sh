#!/bin/bash -l
#SBATCH -q debug
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH -J izhi
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user vbaratham@berkeley.edu
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --output "/global/cscratch1/sd/vbaratha/izhi/runs/%j/stdout.txt"
#SBATCH --error "/global/cscratch1/sd/vbaratha/izhi/runs/%j/stderr.txt"

mkdir runs/${SLURM_JOB_ID}
OUTFILE=runs/${SLURM_JOB_ID}/data.nwb

python run.py --outfile $OUTFILE --create

srun --label -n 128 python param_sweep.py --outfile $OUTFILE

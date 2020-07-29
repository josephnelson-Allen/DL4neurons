#!/bin/bash
#PBS -q celltypes
#PBS -N test_izhi_1
#PBS -m a
#PBS -r n
#PBS -l ncpus=1,mem=1g,walltime=0:05:00
#PBS -o /home/joseph.nelson/DL4neurons/hh2comp_jobtest_1.out
#PBS -j oe
source activate ateam_opt
cd /home/joseph.nelson/DL4neurons
python run.py --model hh_2compartment --dt 0.04 --param-file /home/joseph.nelson/sbi_celltypes/code/hh2comp_params.csv --stim-file /home/joseph.nelson/sbi_celltypes/code/long_square.csv --stim-multiplier 50.0 --outfile /home/joseph.nelson/DL4neurons/hpc_2comp.h5
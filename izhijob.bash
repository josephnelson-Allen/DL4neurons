#!/bin/bash
#PBS -q celltypes
#PBS -N test_izhi_1
#PBS -m a
#PBS -r n
#PBS -l ncpus=1,mem=1g,walltime=0:10:00
#PBS -o /home/joseph.nelson/DL4neurons/izhi_jobtest_1.out
#PBS -j oe
source activate ateam_opt
cd /home/joseph.nelson/DL4neurons
python run.py --model izhi --dt 0.04 --params 0.02 0.2 -65 8 --stim-file /home/joseph.nelson/sbi_celltypes/code/ramp2.csv --stim-multiplier 1 --outfile /home/joseph.nelson/DL4neurons/hpc_izhi2.h5

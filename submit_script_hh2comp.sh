
for p1 in -0.110 -0.90 -0.70 -0.50 -0.30 -0.10 0.10 0.30 0.50 0.70 0.90 0.110 0.130 0.150 0.170 
    do
        jobid="run_"$p1_"hh2comp"
        echo '#!/bin/bash'>subjob.bash
        echo '#PBS -q celltypes'>>subjob.bash
        echo '#PBS -N '${jobid//./-}>>subjob.bash
        echo '#PBS -m a'>>subjob.bash
        echo '#PBS -r n'>>subjob.bash
        echo '#PBS -l ncpus=1,mem=1g,walltime=00:05:00'>>subjob.bash
        echo '#PBS -o /home/joseph.nelson/DL4neurons/'${jobid//./-}'.out'>>subjob.bash
        echo '#PBS -j oe'>>subjob.bash
        echo 'source activate ateam_opt'>>subjob.bash
        echo 'cd /home/joseph.nelson/DL4neurons'>>subjob.bash
        echo 'python run.py --model hh_2compartment --dt 0.04 --param-file /home/joseph.nelson/sbi_celltypes/code/hh2comp_paramset.csv --stim-file /home/joseph.nelson/sbi_celltypes/code/long_square.csv --stim-multiplier '$p1' --outfile hh2comp_'$p1'.h5'>>subjob.bash
        echo '...'
        sleep 1
        wait
        qsub subjob.bash
        echo 'Job: '$jobid' '
    done

#Clean up
rm subjob.bash
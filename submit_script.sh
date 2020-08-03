for p1 in "/home/joseph.nelson/sbi_celltypes/code/ramp.csv" "/home/joseph.nelson/sbi_celltypes/code/ramp2.csv"
do
    for p2 in "/home/joseph.nelson/DL4neurons/hpc_izhi1.h5" "/home/joseph.nelson/DL4neurons/hpc_izhi2.h5"
    do
        jobid="run_"$p1"_"$p2
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
        echo 'python run.py --model izhi --dt 0.04 --params 0.02 0.2 -65 8 --stim-file '$p1' --stim-multiplier 1 --outfile '$p2>>subjob.bash
        echo '...'
        sleep 1
        wait
        qsub subjob.bash
        echo 'Job: '$jobid' '
    done
done

#Clean up
rm subjob.bash
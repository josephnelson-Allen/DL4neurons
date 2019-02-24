from __future__ import print_function

import itertools
import sys
import os
import subprocess

import numpy as np

from stimulus import stims


##################################
##### DISPLAY SLURM ENV VARS #####
##################################

slurm_env_vars = (
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_PROCID",
    "SLURM_JOB_NODELIST",
    "SLURM_NODELIST",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_COUNT",
    "SLURM_ARRAY_TASK_MIN",
    "SLURM_ARRAY_TASK_MAX",
    "SLURM_NTASKS",
)
for var in slurm_env_vars:
    print("{}\t{}".format(var, os.environ.get(var, "")))

num_tasks = int(os.environ.get("SLURM_NTASKS") or 1)

task_i = int(os.environ.get("SLURM_PROCID") or 0)

# CREATE LIST OF PARAM SETS

ndim = 4
nsamples = 10000
nsamples_per_dim = int(nsamples**(1./ndim))

aa = np.linspace(0, 0.05, nsamples_per_dim)
bb = np.linspace(0, 2.0, nsamples_per_dim)
cc = np.linspace(-80, -50, nsamples_per_dim)
dd = np.linspace(0, 20, nsamples_per_dim)

paramsets = list(itertools.product(aa, bb, cc, dd))

# CHOOSE PARAMS AND RUN
params_per_task = (len(paramsets) // num_tasks) + 1
start = params_per_task * task_i
stop = min(params_per_task * (task_i + 1), len(paramsets))

passthru = ' '.join(sys.argv[1:])

print('{} total param sets'.format(len(paramsets)))
print('{} param sets per task'.format(params_per_task))

for a, b, c, d in paramsets[start:stop]:
    if '--print-only' in sys.argv:
        print('a = {}'.format(a))
        print('b = {}'.format(b))
        print('c = {}'.format(c))
        print('d = {}'.format(d))
        print('-'*80)
    else:    
        for stim_type, stim_list in stims.items():
            for i in range(len(stim_list)):
                args = '--a {} --b {} --c {} --d {}'.format(a, b, c, d)
                args += ' --stim-type {} --stim-idx {}'.format(stim_type, i)
                args += ' {}'.format(passthru)
                print("Running {}".format(args))
                subprocess.call('python run.py {}'.format(args), shell=True)

import sys
import os
import subprocess

import numpy as np


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
    print "%s\t%s" % (var, os.environ.get(var, ""))

num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT") or 0)

task_i = int(os.environ.get("SLURM_ARRAY_TASK_ID") or 0)

# CREATE LIST OF PARAM SETS

ndim = 4
nsamples = 10000
nsamples_per_dim = int(nsamples**(1./ndim))

aa = np.linspace(-0.05, 0.05, nsamples_per_dim)
bb = np.linspace(-0.5, 2.0samples_per_dim)
cc = np.linspace(-80, -50, nsamples_per_dim)
dd = np.linspace(-30, 20, nsamples_per_dim)

paramsets = list(itertools.product(aa, bb, cc, dd))

# CHOOSE PARAM
if task_i < len(paramsets):
    a, b, c, d = paramsets[task_i]
else:
    exit()

# RUN

# TODO: Run on all stims
# TODO: Run a few param sets, depending on num_tasks
passthru = ' '.join(sys.argv)
subprocess.call('python --a {a} --b {b} --c {c} --d {d} {addl}'.format(a=a, b=b, c=c, d=d, addl=passthru)

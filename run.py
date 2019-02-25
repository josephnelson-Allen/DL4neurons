from __future__ import print_function

"""
Requires izhi2003a.mod from:
https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=39948

Run this script from the same directory as the compiled izhi2003a.mod

NB: on OSX, run using `pythonw` (first, `conda install python.app`) for plotting to work
"""

import json
import itertools
import logging as log
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import h5py
# from pynwb import NWBFile, NWBHDF5IO, TimeSeries

from stimulus import stims, add_stims

try:
    from mpi4py import MPI
    mpi = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_tasks = comm.Get_size()
except:
    mpi = False
    comm = None
    rank = 0
    n_tasks = 1
    
from neuron import h, gui

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)
    
# redefine NEURON's advance() (runs one timestep) to update the current
h('proc advance() {nrnpython("myadvance()")}')
def myadvance():
    h.cell.Iin = h.stim[int(h.t/h.dt)]
    h.fadvance()

NSAMPLES = 10000
def get_paramsets(nsamples=NSAMPLES):
    """
    Return the full list of parameter values for each run
    """
    ndim = 4
    nsamples_per_dim = int(nsamples**(1./ndim))

    log.info("For {} points in {}-D parameter space, we take {} samples per dimension".format(
        nsamples, ndim, nsamples_per_dim
    ))

    aa = np.linspace(0, 0.05, nsamples_per_dim)
    bb = np.linspace(0, 2.0, nsamples_per_dim)
    cc = np.linspace(-75, -55, nsamples_per_dim)
    dd = np.linspace(0, 20, nsamples_per_dim)
    
    return list(itertools.product(aa, bb, cc, dd))


def get_mpi_idx(nsamples=NSAMPLES):
    params_per_task = (nsamples // n_tasks) + 1
    start = params_per_task * rank
    stop = min(params_per_task * (rank + 1), nsamples)
    log.info("There are {} ranks, so each rank gets {} param sets".format(n_tasks, params_per_task))
    log.info("This rank is processing param sets {} through {}".format(start, stop))

    return start, stop


def get_params_mpi(nsamples=NSAMPLES):
    """
    Get the list of parameter sets for this MPI task
    """
    paramsets = get_paramsets()
    start, stop = get_mpi_idx()

    return paramsets[start:stop], start, stop


def get_stim(stim_type, i):
    # TODO?: variable length stimuli, or determine simulation duration from stimulus length?
    return stims[stim_type][i]


def create_h5(args, nsamples=NSAMPLES):
    """
    Run in serial mode
    """
    # TODO: tstop, rate, and other parameters
    log.info("Creating h5 file and writing param values")

    with h5py.File(args.outfile, 'w') as f:
        # write params
        shape_param = (nsamples,)
        f.create_dataset('a', shape=shape_param, dtype=np.float64)
        f.create_dataset('b', shape=shape_param, dtype=np.float64)
        f.create_dataset('c', shape=shape_param, dtype=np.float64)
        f.create_dataset('d', shape=shape_param, dtype=np.float64)
        for i, (a, b, c, d) in enumerate(get_paramsets()):
            f['a'][i] = a
            f['b'][i] = b
            f['c'][i] = c
            f['d'][i] = d

        # create stim and voltage datasets
        ntimepts = int(args.tstop/args.dt)
        shape_dset = (ntimepts, nsamples)
        for stim_type, stim_list in stims.items():
            for i, stim in enumerate(stim_list):
                dset = '{}_{:02d}'.format(stim_type, i)
                v_name = '{}_v'.format(dset)
                stim_name = '{}_stim'.format(dset)
                f.create_dataset(v_name, shape=shape_dset, dtype=np.float64)
                f.create_dataset(stim_name, data=stim)

    log.info("Done.")

    
def save_h5(args, buf, start, stop):
    log.info("saving into h5")
    dset_name = '{}_{:02d}_v'.format(args.stim_type, args.stim_idx)
    if comm and n_tasks > 1:
        kwargs = {'driver': 'mpio', 'comm': comm}
    else:
        kwargs = {}
    with h5py.File(args.outfile, 'a', **kwargs) as f:
        log.info("opened h5")
        f[dset_name][:, start:stop] = buf
        log.info("saved h5")
    log.info("closed h5")
        

def create_nwb(args):
    log.info("Creating and writing nwb file {}...".format(args.outfile))
    nwb = NWBFile(
        session_description='izhikevich simulation',
        identifier='izhi',
        session_start_time=datetime.now(),
        file_create_date=datetime.now(),
        experimenter='Vyassa Baratham',
        experiment_description='izhikevich simulations for DL',
        session_id='izhi',
        institution='LBL/UCB',
        lab='NSE Lab',
        pharmacology='',
        notes='',
    )
    add_stims(nwb)
    with NWBHDF5IO(args.outfile, 'w') as io:
        io.write(nwb)
    log.info("Done.")


def save_nwb(args, v, a, b, c, d):
    # outfile must exist
    log.info("Saving nwb...")
    
    with NWBHDF5IO(args.outfile, 'a', comm=comm) as io:
        nwb = io.read()
        params_dict = {'a': a, 'b': b, 'c': c, 'd': d}

        stim_str = '{}_{:02d}'.format(args.stim_type, args.stim_idx)
        param_str = hash((a, b, c, d))
        dset_name = '{}__{}'.format(stim_str, param_str)
        dset = TimeSeries(name=dset_name, data=v, description=json.dumps(params_dict),
                          starting_time=0.0, rate=1.0/h.dt)
        nwb.add_acquisition(dset)

        io.write(nwb)
        
    log.info("done.")
    

def plot(args, stim, u, v):
    t_axis = np.linspace(0, len(v)*h.dt, len(v)-1)
    
    if args.plot_stim:
        plt.plot(t_axis, stim)
        plt.show()
    
    if args.plot_v:
        plt.plot(t_axis, v[:-1])
        plt.show()

    if args.plot_u:
        plt.plot(t_axis, u[:-1])
        plt.show()


def simulate(args, a, b, c, d):
    _start = datetime.now()

    # Simulation parameters
    h.tstop = args.tstop
    h.steps_per_ms = 1./args.dt
    h.stdinit() # sets h.dt based on h.steps_per_ms
    ntimepts = int(h.tstop/h.dt)

    if h.dt != args.dt:
        raise ValueError("Invalid choice of dt")

    # Define the cell
    dummy = h.Section()
    cell = h.Izhi2003a(0.5,sec=dummy)
    cell.a = a
    cell.b = b
    cell.c = c
    cell.d = d

    # Define the stimulus
    stim = get_stim(args.stim_type, args.stim_idx)

    # Make the stimulus and cell objects accessible from hoc
    h('objref stim')
    h.stim = stim
    h('objref cell')
    h.cell = cell
    
    # Set up recordings of u and v
    u = h.Vector(ntimepts)
    u.record(cell._ref_u)

    v = h.Vector(ntimepts)
    v.record(cell._ref_V)

    # Run
    log.info("Running simulation for {} ms with dt = {}".format(h.tstop, h.dt))
    log.info("({} total timesteps)".format(ntimepts))

    h.run()

    log.info("Time to simulate: {}".format(datetime.now() - _start))

    # numpy-ify
    u = np.array(u)
    v = np.array(v)

    # Plot
    plot(args, stim, u, v)

    return u, v

def main(args):

    if not any([args.plot_stim, args.plot_u, args.plot_v, args.outfile, args.force]):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")

    if args.param_sweep:
        paramsets, start, stop = get_params_mpi()
    else:
        paramsets = [(args.a, args.b, args.c, args.d)]
        start, stop = 0, 1
        ntimepts = int(args.tstop/args.dt)
        buf = np.zeros(shape=(ntimepts, stop-start), dtype=np.float64)

    # for i, (a, b, c, d) in zip(range(start, stop), paramsets):
    for i, (a, b, c, d) in enumerate(paramsets):
        log.info("About to run a={}, b={}, c={}, d={}".format(a, b, c, d))
        u, v = simulate(args, a, b, c, d)
        buf[:, i] = v[:-1]
            
    # Save to disk
    if args.outfile:
        # save_nwb(args, v, a, b, c, d)
        save_h5(args, buf, start, stop)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help='nwb file to save to. Must exist.')
    parser.add_argument('--create', action='store_true', default=False,
                        help="create the file, store all stimuli, and then exit")
    
    parser.add_argument('--plot-v', action='store_true', default=False)
    parser.add_argument('--plot-u', action='store_true', default=False)
    parser.add_argument('--plot-stim', action='store_true', default=False)
    
    parser.add_argument('--force', action='store_true', default=False,
                        help="make the script run even if you don't plot or save anything")

    parser.add_argument('--param-sweep', action='store_true', default=False,
                        help="run over all values of a,b,c,d")
    
    parser.add_argument('--tstop', type=int, default=152)
    parser.add_argument('--dt', type=float, default=.02)

    parser.add_argument('--a', type=float, default=0.02)
    parser.add_argument('--b', type=float, default=0.2)
    parser.add_argument('--c', type=float, default=-65.)
    parser.add_argument('--d', type=float, default=2.)

    parser.add_argument('--stim-type', type=str, default='ramp')
    parser.add_argument('--stim-idx', '--stim-i', type=int, default=0)

    args = parser.parse_args()

    if args.create:
        create_h5(args)
        # create_nwb(args)
    else:
        main(args)

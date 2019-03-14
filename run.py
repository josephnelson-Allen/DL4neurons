from __future__ import print_function

"""
Run this script from the same directory as the compiled izhi2003a.mod and ca.mod

NB: on OSX, run using `pythonw` (first, `conda install python.app`) for plotting to work
"""

import os
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

DEFAULT_PARAMS = {
    'izhi': (0.02, 0.2, -65., 2.),
    'hh_point_5param': (500, 10, 1.5, .0005, 0.5),
}
NPARAMS = {
    'izhi': 4,
    'hh_point_5param': 5,
}

range_a = (0.01, 0.1)
range_b = (0.1, 0.4)
range_c = (-80, -50)
range_d = (0.5, 5)

range_gnabar = (200, 800)
range_gkbar = (8, 15)
range_gcabar = (1, 2)
range_gl = (0.0004, 0.00055)
range_cm = (0.3, 0.7)


RANGES = {
    'izhi': (range_a, range_b, range_c, range_d),
    'hh_point_5param': (range_gnabar, range_gkbar, range_gcabar, range_gl, range_cm),
}

NSAMPLES = 10000

    
def myadvance():
    idx = int(h.t/h.dt)
    if idx < len(h.stim):
        h.cell.Iin = h.stim[int(h.t/h.dt)]
    h.fadvance()

    
def _rangeify(data, _range):
    return data * (_range[1] - _range[0]) + _range[0]


def clean_params(args):
    """
    convert to float, use defaults where requested
    """
    defaults = DEFAULT_PARAMS[args.model]
    if args.params:
        return [float(x if x != 'rand' else 'inf') if x != 'def' else default
                for (x, default) in zip(args.params, defaults)]
    else:
        return [float('inf')] * len(defaults)

    
def get_random_params(args, n=1):
    ranges = RANGES[args.model]
    ndim = len(ranges)
    rand = np.random.rand(n, ndim)
    params = clean_params(args)
    for i, (_range, param) in enumerate(zip(ranges, params)):
        # Default params swapped in by clean_params()
        if param == float('inf'):
            rand[:, i] = _rangeify(rand[:, i], _range)
        else:
            rand[:, i] = np.array([param] * n)
    return rand

        
def get_mpi_idx(args, nsamples=NSAMPLES):
    params_per_task = (nsamples // n_tasks) + 1
    start = params_per_task * rank
    stop = min(params_per_task * (rank + 1), nsamples)
    if args.num:
        stop = min(stop, args.num)
    log.info("There are {} ranks, so each rank gets {} param sets".format(n_tasks, params_per_task))
    log.info("This rank is processing param sets {} through {}".format(start, stop))

    return start, stop


def get_params_mpi(nsamples=NSAMPLES):
    """
    Get the list of parameter sets for this MPI task
    """
    paramsets = get_paramsets(nsamples=nsamples)
    start, stop = get_mpi_idx(nsamples=nsamples)

    return paramsets[start:stop], start, stop


multiplier = {
    'Ramp_0p5.csv': 30.0,
    'Step_0p2.csv': 30.0,
    'chirp_05.csv': 10.0,
    'chirp_damp.csv': 15.0,
    'chirp_damp_8k.csv': 15.0,
    'chirp_damp_10k.csv': 15.0,
    'he_1i_1.csv': 20.0,
}
def get_stim(args):
    # TODO?: variable length stimuli, or determine simulation duration from stimulus length?
    if args.stim_file:
        stim_fn = os.path.basename(args.stim_file)
        stim = (np.genfromtxt(args.stim_file, dtype=np.float64) + args.stim_dc_offset) * multiplier[stim_fn]
    else:
        stim = stims[args.stim_type][args.stim_idx]

    # silence = np.zeros(int(args.silence/args.dt))
    # return np.concatenate([silence, stim, silence])
    
    return stim


def attach_stim(args):
    if args.model == 'izhi':
        # redefine NEURON's advance() (runs one timestep) to update the current
        h('proc advance() {nrnpython("myadvance()")}')
    elif args.model == 'hh_point_5param':
        # Use an IClamp object
        h('objref clamp')
        clamp = h.IClamp(0.5)
        clamp.delay = 0
        clamp.dur = h.tstop
        h.clamp = clamp

        h('objref stimvals')
        stimvals = h.Vector().from_python(h.stim)
        h.stimvals = stimvals
        stimvals.play("clamp.amp = $1", h.dt)
    else:
        raise ValueError("choose 'izhi' or 'hh_point_5param'")


def create_h5(args, nsamples=NSAMPLES):
    """
    Run in serial mode
    """
    # TODO: tstop, rate, and other parameters
    log.info("Creating h5 file")
    with h5py.File(args.outfile, 'w') as f:
        # write params
        ndim = NPARAMS[args.model]
        f.create_dataset('phys_par', shape=(nsamples, ndim))
        f.create_dataset('norm_par', shape=(nsamples, ndim), dtype=np.float64)

        # write param range
        phys_par_range = np.stack(RANGES[args.model])
        f.create_dataset('phys_par_range', data=phys_par_range, dtype=np.float64)

        # create stim and voltage datasets
        stim = get_stim(args)
        ntimepts = int(args.tstop/args.dt)
        f.create_dataset('voltages', shape=(nsamples, ntimepts), dtype=np.float64)
        f.create_dataset('stim', data=stim)
    log.info("Done.")


def _normalize(args, data, minmax=1):
    nsamples = data.shape[0]
    mins = np.array([tup[0] for tup in RANGES[args.model]])
    mins = np.tile(mins, (nsamples, 1)) # stacked to same shape as input
    ranges = np.array([tup[1]-tup[0] for tup in RANGES[args.model]])
    ranges = np.tile(ranges, (nsamples, 1))

    return 2*minmax * ( (data - mins)/ranges ) - minmax

    
def save_h5(args, buf, params, start, stop):
    log.info("saving into h5")
    if comm and n_tasks > 1:
        log.debug("using parallel")
        kwargs = {'driver': 'mpio', 'comm': comm}
    else:
        log.debug("using serial")
        kwargs = {}
    with h5py.File(args.outfile, 'a', **kwargs) as f:
        log.debug("opened h5")
        log.debug(str(params))
        f['phys_par'][start:stop, :] = params
        mins, maxes = np.min(params, axis=0), np.max(params, axis=0)
        f['norm_par'][start:stop, :] = _normalize(args, params)
        f['voltages'][start:stop, :] = buf
        log.debug("saved h5")
    log.debug("closed h5")


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
    

def plot(args, data, stim):
    ntimepts = int(h.tstop/h.dt)
    t_axis = np.linspace(0, ntimepts*h.dt, ntimepts)
    if args.plot or args.plot_v:
        plt.plot(t_axis, data['v'][:ntimepts], label='V_m')
    if args.plot or args.plot_stim:
        plt.plot(t_axis, stim[:ntimepts], label='stim')
    if args.plot or args.plot_ina:
        plt.plot(t_axis, data['ina'][:ntimepts] * 100, label='i_na*100')
    if args.plot or args.plot_ik:
        plt.plot(t_axis, data['ik'][:ntimepts] * 100, label='i_k*100')
    if args.plot or args.plot_ica:
        plt.plot(t_axis, data['ica'][:ntimepts] * 100, label='i_ca*100')
    if args.plot or args.plot_i_cap:
        plt.plot(t_axis, data['i_cap'][:ntimepts] * 100, label='i_cap*100')
    if args.plot or args.plot_i_leak:
        plt.plot(t_axis, data['i_leak'][:ntimepts] * 100, label='i_leak*100')

    if not args.no_legend:
        plt.legend()
        
    plt.show()
    
# def silence_spikes(v, args):
#     npts = int(args.silence/args.dt)
#     return max(v[:npts]) > 0


def simulate(args, params):
    _start = datetime.now()

    h.celsius = args.celsius

    # Simulation parameters
    h.tstop = args.tstop
    h.steps_per_ms = 1./args.dt
    h.stdinit() # sets h.dt based on h.steps_per_ms
    ntimepts = int(h.tstop/h.dt)

    if h.dt != args.dt:
        raise ValueError("Invalid choice of dt")

    hoc_vectors = {
        'v': h.Vector(ntimepts),
        'ina': h.Vector(ntimepts),
        'ik': h.Vector(ntimepts),
        'ica': h.Vector(ntimepts),
        'i_leak': h.Vector(ntimepts),
        'i_cap': h.Vector(ntimepts),
    }
    
    # Define the cell
    # This doesn't work when done in a separate function, for some reason
    # I think 'dummy' needs to stay in scope?
    if args.model == 'izhi':
        dummy = h.Section()
        cell = h.Izhi2003a(0.5,sec=dummy)
        
        a, b, c, d = params
        cell.a = a
        cell.b = b
        cell.c = c
        cell.d = d
        
        hoc_vectors['v'].record(cell._ref_V) # Capital V because it's not the real membrane voltage
    elif args.model == 'hh_point_5param':
        cell = h.Section()
        cell.insert('na')
        cell.insert('kv')
        cell.insert('ca')
        cell.insert('pas')

        gnabar, gkbar, gcabar, gl, cm = params
        cell(0.5).na.gbar = gnabar
        cell(0.5).kv.gbar = gkbar
        cell(0.5).ca.gbar = gcabar
        cell(0.5).pas.g = gl
        cell.cm = cm

        hoc_vectors['v'].record(cell(0.5)._ref_v)
        hoc_vectors['ina'].record(cell(0.5)._ref_ina)
        hoc_vectors['ica'].record(cell(0.5)._ref_ica)
        hoc_vectors['ik'].record(cell(0.5)._ref_ik)
        hoc_vectors['i_cap'].record(cell(0.5).pas._ref_i)
        hoc_vectors['i_cap'].record(cell(0.5)._ref_i_cap)
    else:
        raise ValueError("choose 'izhi' or 'hh_point_5param'")

    # Define the stimulus
    stim = get_stim(args)

    # Make the stimulus and cell objects accessible from hoc
    h('objref stim')
    h.stim = stim
    h('objref cell')
    h.cell = cell

    # attach the stim to the cell
    attach_stim(args)

    # Run
    log.debug("Running simulation for {} ms with dt = {}".format(h.tstop, h.dt))
    log.debug("({} total timesteps)".format(ntimepts))

    h.run()

    log.debug("Time to simulate: {}".format(datetime.now() - _start))

    # numpy-ify
    data = {k: np.array(v) for k, v in hoc_vectors.items()}

    # Plot
    plot(args, data, stim)
        
    return data

def main(args):

    if not any([args.plot, args.plot_v, args.plot_stim, args.plot_ina, args.plot_ik,
                args.plot_ica, args.outfile, args.force]):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")
    
    if args.stim_file:
        log.info("not using --tstop because --stim-file was specified")
        args.tstop = len(get_stim(args))

    if args.param_file:
        all_paramsets = np.genfromtxt(args.param_file, dtype=np.float64)
        start, stop = get_mpi_idx(args, len(all_paramsets))
        if args.num and start > args.num:
            return
        paramsets = all_paramsets[start:stop, :]
    elif args.num:
        start, stop = get_mpi_idx(args, args.num)
        paramsets = get_random_params(args, n=stop-start)
    elif args.params not in (None, [None]):
        paramsets = np.atleast_2d(np.array(args.params))
        start, stop = 0, 1
    else:
        log.info("Cell parameters not specified, running with default parameters")
        paramsets = [ DEFAULT_PARAMS[args.model] ]
        start, stop = 0, 1

    # bad_params = []
    ntimepts = int(args.tstop/args.dt)
    buf = np.zeros(shape=(stop-start, ntimepts), dtype=np.float64)

    for i, params in enumerate(paramsets):
        if args.print_every and i % args.print_every == 0:
            log.info("Processed {} samples".format(i))
        log.debug("About to run with params = {}".format(params))
        data = simulate(args, params)
        buf[i, :] = data['v'][:-1]
        
    # Save to disk
    if args.outfile:
        # save_nwb(args, v, a, b, c, d)
        save_h5(args, buf, paramsets, start, stop)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', choices=['izhi', 'hh_point_5param'], default='hh_point_5param')
    parser.add_argument('--celsius', type=float, default=33)

    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help='nwb file to save to. Must exist unless --create is passed')
    parser.add_argument('--create', action='store_true', default=False,
                        help="create the file, store all stimuli, and then exit")
    parser.add_argument('--create-params', action='store_true', default=False,
                        help="create the params file (--param-file) and exit. Must use with --num")

    parser.add_argument('--plot', action='store_true', default=False,
                        help="plot everything")
    parser.add_argument('--plot-stim', action='store_true', default=False,
                        help="plot stimulus")
    parser.add_argument('--plot-v', action='store_true', default=False,
                        help="plot voltage")
    parser.add_argument('--plot-ina', action='store_true', default=False,
                        help="plot sodium current")
    parser.add_argument('--plot-ik', action='store_true', default=False,
                        help="plot potassium current")
    parser.add_argument('--plot-ica', action='store_true', default=False,
                        help="plot calcium current")
    parser.add_argument('--plot-i-cap', '--plot-icap', action='store_true', default=False,
                        help="plot capacitive current")
    parser.add_argument('--plot-i-leak', '--plot-ileak', action='store_true', default=False,
                        help="plot leak current")
    parser.add_argument('--no-legend', action='store_true', default=False,
                        help="do not display the legend on the plot")
    
    parser.add_argument('--force', action='store_true', default=False,
                        help="make the script run even if you don't plot or save anything")

    parser.add_argument('--tstop', type=int, default=200)
    parser.add_argument('--dt', type=float, default=.02)

    # parser.add_argument('--silence', type=int, default=0,
    #                     help="amount of pre/post-stim silence (ms)")

    # CHOOSE PARAMETERS
    parser.add_argument('--num', type=int, default=None, required=False,
                        help="number of param values to choose. Will choose randomly. " + \
                        "See --params. When multithreaded, this is the total number over all ranks")
    parser.add_argument('--params', type=str, nargs='+', default=None,
                        help='When used with --num, fixes the value of some params. To indicate ' + \
                        'that a param should not be held fixed, set it to "rand". ' + \
                        'to use the default value, use "def"' + \
                        'eg to use the default 1st param, random 2nd param, ' + \
                        'and specific values 3.0 and 4.0 for the 3rd and 4th params, use "def inf 3.0 4.0"'
    )
    parser.add_argument('--param-file', '--params-file', type=str, required=False, default=None)

    # CHOOSE STIMULUS
    parser.add_argument('--stim-type', type=str, default=None)
    parser.add_argument('--stim-idx', '--stim-i', type=int, default=0)
    parser.add_argument('--stim-file', type=str, required=False, default='stims/chirp_damp_10k.csv',
                        help="Use a csv for the stimulus file, overrides --stim-type and --stim-idx and --tstop")
    parser.add_argument('--stim-dc-offset', type=float, default=0.0)

    parser.add_argument('--print-every', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.params and not args.num:
        args.num = 1

    log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG if args.debug else log.INFO)

    # args.tstop += 2 * args.silence

    if args.create:
        create_h5(args, nsamples=(args.num or NSAMPLES))
        # create_nwb(args)
    elif args.create_params:
        np.savetxt(args.param_file, get_random_params(args, n=args.num))
    else:
        main(args)

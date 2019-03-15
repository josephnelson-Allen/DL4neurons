from __future__ import print_function

import os
import json
import logging as log
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import h5py
# from pynwb import NWBFile, NWBHDF5IO, TimeSeries

from stimulus import stims, add_stims
import models

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

ALL_MODELS = (
)
MODELS_BY_NAME = {
    'izhi': models.Izhi,
    'hh_point_5param': models.HHPoint5Param,
    'hh_ball_stick_7param': models.HHBallStick7Param,
}

    
def _rangeify(data, _range):
    return data * (_range[1] - _range[0]) + _range[0]


def clean_params(args):
    """
    convert to float, use defaults where requested
    """
    defaults = MODELS_BY_NAME[args.model].DEFAULT_PARAMS
    if args.params:
        return [float(x if x != 'rand' else 'inf') if x != 'def' else default
                for (x, default) in zip(args.params, defaults)]
    else:
        return [float('inf')] * len(defaults)

    
def get_random_params(args, n=1):
    ranges = MDOELS_BY_NAME[args.model].PARAM_RANGES
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

        
def get_mpi_idx(args, nsamples):
    params_per_task = (nsamples // n_tasks) + 1
    start = params_per_task * rank
    stop = min(params_per_task * (rank + 1), nsamples)
    if args.num:
        stop = min(stop, args.num)
    log.info("There are {} ranks, so each rank gets {} param sets".format(n_tasks, params_per_task))
    log.info("This rank is processing param sets {} through {}".format(start, stop))

    return start, stop


def get_stim(args):
    # TODO?: variable length stimuli, or determine simulation duration from stimulus length?
    if args.stim_file:
        stim_fn = os.path.basename(args.stim_file)
        multiplier = args.stim_multiplier or (.12 if args.model == 'hh_ball_stick_7param' else 15.0)
        log.debug("Stim multiplier = {}".format(multiplier))
        return (np.genfromtxt(args.stim_file, dtype=np.float64) + args.stim_dc_offset) * multiplier
    else:
        return stims[args.stim_type][args.stim_idx]


def create_h5(args, nsamples):
    """
    Run in serial mode
    """
    # TODO: tstop, rate, and other parameters
    log.info("Creating h5 file")
    with h5py.File(args.outfile, 'w') as f:
        # write params
        ndim = len(RANGES[args.model])
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
    mins = np.array([tup[0] for tup in MODELS_BY_NAME[args.model].PARAM_RANGES])
    mins = np.tile(mins, (nsamples, 1)) # stacked to same shape as input
    ranges = np.array([_max - _min for (_min, _max) in MODELS_BY_NAME[args.model].PARAM_RANGES])
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


def plot(args, data, stim):
    ntimepts = int(h.tstop/h.dt)
    t_axis = np.linspace(0, ntimepts*h.dt, ntimepts)
    
    plt.figure(figsize=(10, 5))
    plt.xlabel('Time (ms)')

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


def main(args):
    if not any([args.plot, args.plot_v, args.plot_stim, args.plot_ina, args.plot_ik,
                args.plot_ica, args.outfile, args.force]):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")
    
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
        paramsets = [ MODELS_BY_NAME[args.model].DEFAULT_PARAMS ]
        start, stop = 0, 1

    stim = get_stim(args)
    buf = np.zeros(shape=(stop-start, len(stim)), dtype=np.float64)

    for i, params in enumerate(paramsets):
        if args.print_every and i % args.print_every == 0:
            log.info("Processed {} samples".format(i))
        log.debug("About to run with params = {}".format(params))

        model = MODELS_BY_NAME[args.model](*params, log=log)
        data = model.simulate(stim, args.dt)
        buf[i, :] = data['v'][:-1]

        plot(args, data, stim)
        
    # Save to disk
    if args.outfile:
        save_h5(args, buf, paramsets, start, stop)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', choices=['izhi', 'hh_point_5param', 'hh_ball_stick_7param'],
                        default='hh_ball_stick_7param')
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

    # CHOOSE PARAMETERS
    parser.add_argument('--num', type=int, default=None,
                        help="number of param values to choose. Will choose randomly. " + \
                        "See --params. When multithreaded, this is the total number over all ranks")
    parser.add_argument('--params', type=str, nargs='+', default=None,
                        help='When used with --num, fixes the value of some params. To indicate ' + \
                        'that a param should not be held fixed, set it to "rand". ' + \
                        'to use the default value, use "def"' + \
                        'eg to use the default 1st param, random 2nd param, ' + \
                        'and specific values 3.0 and 4.0 for the 3rd and 4th params, use "def inf 3.0 4.0"'
    )
    parser.add_argument('--param-file', '--params-file', type=str, default=None)

    # CHOOSE STIMULUS
    parser.add_argument('--stim-type', type=str, default=None)
    parser.add_argument('--stim-idx', '--stim-i', type=int, default=0)
    parser.add_argument('--stim-file', type=str, default='stims/chirp_damp_16k_v1.csv',
                        help="Use a csv for the stimulus file, overrides --stim-type and --stim-idx and --tstop")
    parser.add_argument('--stim-dc-offset', type=float, default=0.0)
    parser.add_argument('--stim-multiplier', type=float, default=None)

    parser.add_argument('--print-every', type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.params and not args.num:
        args.num = 1

    log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG if args.debug else log.INFO)

    if args.create:
        create_h5(args, args.num)
    elif args.create_params:
        np.savetxt(args.param_file, get_random_params(args, n=args.num))
    else:
        main(args)

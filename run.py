from __future__ import print_function

import os
import json
import logging as log
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import h5py

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

MODELS_BY_NAME = models.MODELS_BY_NAME

def _rangeify(data, _range):
    return data * (_range[1] - _range[0]) + _range[0]


def clean_params(args):
    """
    convert to float, use defaults where requested
    """
    defaults = MODELS_BY_NAME[args.model].DEFAULT_PARAMS
    if args.params:
        assert len(args.params) == len(defaults)
        return [float(x if x != 'rand' else 'inf') if x != 'def' else default
                for (x, default) in zip(args.params, defaults)]
    else:
        return [float('inf')] * len(defaults)


def report_random_params(args, params):
    param_names = MODELS_BY_NAME[args.model].PARAM_NAMES
    for param, name in zip(params, param_names):
        if param == float('inf'):
            log.info("Using random values for '{}'".format(name))

    
def get_random_params(args, n=1):
    ranges = MODELS_BY_NAME[args.model].PARAM_RANGES
    ndim = len(ranges)
    rand = np.random.rand(n, ndim)
    params = clean_params(args)
    report_random_params(args, params)
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


def get_stim(args, mult=None):
    stim_fn = os.path.basename(args.stim_file)
    multiplier = mult or args.stim_multiplier or MODELS_BY_NAME[args.model].STIM_MULTIPLIER
    log.debug("Stim multiplier = {}".format(multiplier))
    return (np.genfromtxt(args.stim_file, dtype=np.float64) * multiplier) + args.stim_dc_offset


def _qa(args, trace, thresh=20):
    trace = trace[:-1]
    # stim = get_stim(args, mult=1)
    # hyp_trace = trace[stim == -1.0]
    # main_trace = trace[(stim != -1) & (stim != 0)]
    # hyp_trace = np.concatenate([trace[4000:5400], trace[20500:22000]])
    hyp_trace = trace[4000:5400]
    main_trace = trace[6001:16999]

    hyp_crossings = np.diff( (hyp_trace > thresh).astype('int') )
    main_crossings = np.diff( (main_trace > thresh).astype('int') )
    num_hyp_crossings = np.sum(hyp_crossings == 1)
    num_main_crossings = np.sum(main_crossings == 1)

    # print(num_hyp_crossings, num_main_crossings)
    
    return (num_hyp_crossings == 0) and (num_main_crossings > 0) and (num_main_crossings <= 30)


def create_h5(args, nsamples):
    """
    Run in serial mode
    """
    log.info("Creating h5 file {}".format(args.outfile))
    with h5py.File(args.outfile, 'w') as f:
        # write params
        ndim = len(MODELS_BY_NAME[args.model].PARAM_RANGES)
        f.create_dataset('phys_par', shape=(nsamples, ndim), dtype=np.float64)
        f.create_dataset('norm_par', shape=(nsamples, ndim), dtype=np.float64)

        # write param range
        phys_par_range = np.stack(MODELS_BY_NAME[args.model].PARAM_RANGES)
        f.create_dataset('phys_par_range', data=phys_par_range, dtype=np.float64)

        # create stim, qa, and voltage datasets
        stim = get_stim(args)
        ntimepts = len(stim)
        f.create_dataset('voltages', shape=(nsamples, ntimepts), dtype=np.float64)
        f.create_dataset('binQA', shape=(nsamples,), dtype=np.float64)
        f.create_dataset('stim', data=stim)
    log.info("Done.")


def _normalize(args, data, minmax=1):
    nsamples = data.shape[0]
    mins = np.array([tup[0] for tup in MODELS_BY_NAME[args.model].PARAM_RANGES])
    mins = np.tile(mins, (nsamples, 1)) # stacked to same shape as input
    ranges = np.array([_max - _min for (_min, _max) in MODELS_BY_NAME[args.model].PARAM_RANGES])
    ranges = np.tile(ranges, (nsamples, 1))

    return 2*minmax * ( (data - mins)/ranges ) - minmax

    
def save_h5(args, buf, qa, params, start, stop):
    log.info("saving into h5 file {}".format(args.outfile))
    if comm and n_tasks > 1:
        log.debug("using parallel")
        kwargs = {'driver': 'mpio', 'comm': comm}
    else:
        log.debug("using serial")
        kwargs = {}
    with h5py.File(args.outfile, 'a', **kwargs) as f:
        log.debug("opened h5")
        log.debug(str(params))
        f['voltages'][start:stop, :] = buf
        f['binQA'][start:stop] = qa
        if not args.blind:
            f['phys_par'][start:stop, :] = params
            f['norm_par'][start:stop, :] = _normalize(args, params)
        log.info("saved h5")
    log.info("closed h5")


def plot(args, data, stim):
    if args.plot is not None:
        ntimepts = len(stim)
        t_axis = np.linspace(0, ntimepts*h.dt, ntimepts)

        plt.figure(figsize=(10, 5))
        plt.xlabel('Time (ms)')

        if args.plot == [] or 'v' in args.plot:
            plt.plot(t_axis, data['v'][:ntimepts], label='V_m')
        if args.plot == [] or 'v_dend' in args.plot:
            plt.plot(t_axis, data['v_dend'][:ntimepts], label='v_dend')
        if args.plot == [] or 'stim' in args.plot:
            plt.plot(t_axis, stim[:ntimepts], label='stim')
        if args.plot == [] or 'ina' in args.plot:
            plt.plot(t_axis, data['ina'][:ntimepts] * 100, label='i_na*100')
        if args.plot == [] or 'ik' in args.plot:
            plt.plot(t_axis, data['ik'][:ntimepts] * 100, label='i_k*100')
        if args.plot == [] or 'ica' in args.plot:
            plt.plot(t_axis, data['ica'][:ntimepts] * 100, label='i_ca*100')
        if args.plot == [] or 'i_cap' in args.plot:
            plt.plot(t_axis, data['i_cap'][:ntimepts] * 100, label='i_cap*100')
        if args.plot == [] or 'i_leak' in args.plot:
            plt.plot(t_axis, data['i_leak'][:ntimepts] * 100, label='i_leak*100')

        if not args.no_legend:
            plt.legend()

        plt.show()


def add_qa(args):
    if comm and n_tasks > 1:
        log.debug("using parallel")
        kwargs = {'driver': 'mpio', 'comm': comm}
    else:
        log.debug("using serial")
        kwargs = {}
        
    start, stop = get_mpi_idx(args, args.num)
        
    with h5py.File(args.outfile, 'r', **kwargs) as f:
        v = f['voltages'][start:stop, :]

    qa = np.zeros(stop-start)

    for i in range(start, stop):
        if args.print_every and i % args.print_every == 0:
            log.info("done {}".format(i))
        qa[i] = _qa(v[i, :])

    with h5py.File(args.outfile, 'a', **kwargs) as f:
        f.create_dataset('qa', shape=(args.num,))
        f['qa'][start:stop] = qa

    log.info("done")


def lock_params(args, paramsets):
    assert len(args.locked_params) % 2 == 0
    
    paramnames = MODELS_BY_NAME[args.model].PARAM_NAMES
    nsets = len(args.locked_params)//2
    
    targets = [args.locked_params[i*2] for i in range(nsets)]
    sources = [args.locked_params[i*2+1] for i in range(nsets)]

    for source, target in zip(sources, targets):
        source_i = paramnames.index(source)
        target_i = paramnames.index(target)
        paramsets[:, target_i] = paramsets[:, source_i]


def main(args):
    if (not args.outfile) and (not args.force) and (args.plot is None):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")

    if args.create:
        if not args.num:
            raise ValueError("Must pass --num when creating h5 file")
        create_h5(args, args.num)
        exit()

    if args.create_params:
        np.savetxt(args.param_file, get_random_params(args, n=args.num))
        exit()

    if args.add_qa:
        add_qa(args)
        exit()
    
    if args.blind and not args.param_file:
        raise ValueError("Must pass --param-file with --blind")

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
        paramsets = np.atleast_2d(MODELS_BY_NAME[args.model].DEFAULT_PARAMS)
        start, stop = 0, 1

    lock_params(args, paramsets)

    stim = get_stim(args)
    buf = np.zeros(shape=(stop-start, len(stim)), dtype=np.float64)
    qa = np.zeros(stop-start)

    for i, params in enumerate(paramsets):
        if args.print_every and i % args.print_every == 0:
            log.info("Processed {} samples".format(i))
        log.debug("About to run with params = {}".format(params))

        model = MODELS_BY_NAME[args.model](*params, log=log, celsius=args.celsius)
        data = model.simulate(stim, args.dt)
        buf[i, :] = data['v'][:-1]
        qa[i] = _qa(args, data['v'])

        plot(args, data, stim)
        
    # Save to disk
    if args.outfile:
        save_h5(args, buf, qa, paramsets, start, stop)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', choices=MODELS_BY_NAME.keys(),
                        default='hh_ball_stick_7param')
    parser.add_argument('--celsius', type=float, default=37)
    parser.add_argument('--dt', type=float, default=.02)

    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help='nwb file to save to. Must exist unless --create is passed')
    parser.add_argument('--create', action='store_true', default=False,
                        help="create the file, store all stimuli, and then exit " \
                        + "(useful for writing to the file from multiple ranks)"
    )
    parser.add_argument(
        '--create-params', action='store_true', default=False,
        help="create the params file (--param-file) and exit. Must use with --num"
    )
    parser.add_argument('--add-qa', action='store_true', default=False)

    parser.add_argument(
        '--plot', nargs='*',
        choices=['v', 'stim', 'ina', 'ica', 'ik', 'i_leak', 'i_cap', 'v_dend'],
        default=None,
        help="--plot w/ no arguments: plot everything. --plot [args]: print the given variables"
    )
    parser.add_argument('--no-legend', action='store_true', default=False,
                        help="do not display the legend on the plot")
    
    parser.add_argument(
        '--force', action='store_true', default=False,
        help="make the script run even if you don't plot or save anything"
    )

    # CHOOSE PARAMETERS
    parser.add_argument(
        '--num', type=int, default=None,
        help="number of param values to choose. Will choose randomly. " + \
        "See --params. When multithreaded, this is the total number over all ranks"
    )
    parser.add_argument(
        '--params', type=str, nargs='+', default=None,
        help='When used with --num, fixes the value of some params. To indicate ' + \
        'that a param should not be held fixed, set it to "rand". ' + \
        'to use the default value, use "def"' + \
        'eg to use the default 1st param, random 2nd param, ' + \
        'and specific values 3.0 and 4.0 for the 3rd and 4th params, use "def inf 3.0 4.0"'
    )
    parser.add_argument('--param-file', '--params-file', type=str, default=None)
    parser.add_argument('--blind', action='store_true', default=False,
                        help='do not save parameter values in the output nwb. ' + \
                        'You better have saved them using --param-file')

    # CHOOSE STIMULUS
    parser.add_argument(
        '--stim-file', type=str, default='stims/chirp23a.csv',
        help="Use a csv for the stimulus file, overrides --stim-type and --stim-idx and --tstop")
    parser.add_argument(
        '--stim-dc-offset', type=float, default=0.0,
        help="apply a DC offset to the stimulus (shift it). Happens after --stim-multiplier"
    )
    parser.add_argument(
        '--stim-multiplier', type=float, default=None,
        help="scale the stimulus amplitude by this factor. Happens before --stim-dc-offset"
    )

    parser.add_argument('--print-every', type=int, default=1000)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--locked-params', '--lock-params', type=str, nargs='+', default=[])
    
    args = parser.parse_args()

    log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG if args.debug else log.INFO)

    main(args)

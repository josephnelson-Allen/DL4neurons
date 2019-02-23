from argparse import ArgumentParser
from datetime import datetime

from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt

from stimulus import stims

"""
Requires izhi2003a.mod from:
https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=39948

Run this script from the same directory as the compiled izhi2003a.mod

NB: on OSX, run using `pythonw` (first, `conda install python.app`) for plotting to work
"""

# redefine NEURON's advance() (runs one timestep) to update the current
h('proc advance() {nrnpython("myadvance()")}')
def myadvance():
    h.cell.Iin = h.stim[int(h.t/h.dt)]
    h.fadvance()
    

def get_stim(stim_type, i):
    # TODO?: variable length stimuli, or determine simulation duration from stimulus length?
    return stims[stim_type][i]


def save_nwb(outfile, stim, v):
    pass


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


def main(args):

    if not any([args.plot_stim, args.plot_u, args.plot_v, args.outfile, args.force]):
        raise ValueError("You didn't choose to plot or save anything. "
                         + "Pass --force to continue anyways")
    
    start = datetime.now()

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
    cell.a = args.a
    cell.b = args.b
    cell.c = args.c
    cell.d = args.d

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
    print("Running simulation for {} ms with dt = {}".format(h.tstop, h.dt))
    print("({} total timesteps)".format(ntimepts))

    h.run()

    print("Time to simulate: {}".format(datetime.now() - start))

    # numpy-ify
    u = np.array(u)
    v = np.array(v)

    # Save to nwb
    if args.outfile:
        save_nwb(args.outfile, stim, v)

    # Plot
    plot(args, stim, u, v)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--outfile', type=str, required=False, default=None,
                        help='nwb file to save to')
    parser.add_argument('--plot-v', action='store_true', default=False)
    parser.add_argument('--plot-u', action='store_true', default=False)
    parser.add_argument('--plot-stim', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False,
                        help="make the script run even if you don't plot or save anything")
    
    parser.add_argument('--tstop', type=int, default=152)
    parser.add_argument('--dt', type=float, default=.02)

    parser.add_argument('--a', type=float, default=0.02)
    parser.add_argument('--b', type=float, default=0.2)
    parser.add_argument('--c', type=float, default=-65.)
    parser.add_argument('--d', type=float, default=2.)

    parser.add_argument('--stim-type', type=str, default='ramp')
    parser.add_argument('--stim-idx', '--stim-i', type=int, default=0)

    args = parser.parse_args()

    main(args)

""" 
Tools for comparing predicted/actual voltage traces
"""
import logging as log
import itertools
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import h5py
import numpy as np
# from scipy.stats import entropy

from models import MODELS_BY_NAME

import pyspike

from neuron import h, gui


def _qa(args, trace, thresh=10):
    return np.sum(np.diff( (trace > thresh).astype('int') ) == 1)

def normalize(vals, minmax=1, modelname='izhi'):
    model_cls = MODELS_BY_NAME[modelname]
    mins = np.array([tup[0] for tup in model_cls.PARAM_RANGES])
    ranges = np.array([_max - _min for (_min, _max) in model_cls.PARAM_RANGES])
    return 2*minmax * ( (vals - mins)/ranges ) - minmax


def _rangeify(data, _range):
    return (data + 1) * (_range[1] - _range[0])/2.0 + _range[0]


def data_for(modelname, stim, *params, dt=0.02):
    """
    Run the simulation and get data back
    """
    model_cls = MODELS_BY_NAME[modelname]
    params = [_rangeify(x, _range) for x, _range in zip(params, model_cls.PARAM_RANGES)]
    model = model_cls(*params, log=log, celsius=37)

    return model.simulate(stim, dt)


def iter_trials(modelname, answersfile, stim=None):
    """
    Iterate over tuples of true, predicted params and voltage traces of both
    """
    ranges = MODELS_BY_NAME[modelname].PARAM_RANGES
    with h5py.File(answersfile) as infile:
        for phys_pred, unit_pred, unit_truth, trace_truth in zip(infile['physPred2D'], infile['unitPred2D'], infile['unitTruth2D'], infile['trace2D']):
            # TODO: cut off traces at correct bins
            if len(trace_truth) == 9000:
                trace_truth = data_for(modelname, stim, *unit_truth)['v']
            trace_pred = data_for(modelname, stim, *unit_pred)['v']

            yield unit_truth, unit_pred, trace_truth, trace_pred


def _similarity(truth_v, predicted_v, method='isi', thresh=10):
    truth_spikes = np.diff( (truth_v > thresh).astype('int') )
    predicted_spikes = np.diff( (predicted_v > thresh).astype('int') )

    truth_spike_times = np.where(truth_spikes > 0)[0]
    predicted_spike_times = np.where(predicted_spikes > 0)[0]

    truth_spike_train = pyspike.SpikeTrain(truth_spike_times, 5500, 14500)
    predicted_spike_train = pyspike.SpikeTrain(predicted_spike_times, 5500, 14500)

    if method == 'isi':
        return np.abs(pyspike.isi_distance(truth_spike_train, predicted_spike_train))
    else:
        raise ValueError("unknown similarity metric")


def similarity_heatmaps(modelname, answersfile, similarity_measure='isi', stim=None):
    if not stim:
        stim = np.genfromtxt('stims/chirp23a.csv')
        stim *= MODELS_BY_NAME[modelname].STIM_MULTIPLIER
    
    binsize = 0.05
    nbins = int(2/binsize)

    model_cls = MODELS_BY_NAME[modelname]
    nparam = len(model_cls.PARAM_RANGES)

    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    # pair -> bins to plot from
    to_plot = {
        (0, 1): {(10, 10):[None, None], (30, 10):[None, None], (30, 30):[None, None] },
        (1, 2): { (10, 10):[None, None], (30, 10):[None, None], (30, 30):[None, None] },
        (2, 3): { (10, 10):[None, None], (30, 10):[None, None], (30, 30):[None, None] },
        (3, 0): { (10, 10):[None, None], (30, 10):[None, None], (30, 30):[None, None] },
    }

    binned_similarities = {pair: defaultdict(lambda: defaultdict(list)) for pair in pairs}
    for i, (truth, prediction, truth_v, predicted_v) in enumerate(iter_trials(modelname, answersfile, stim=stim)):
        bin_idxs = [int((param+1) / binsize) for param in truth]

        similarity = _similarity(truth_v, predicted_v, method='isi')

        # if similarity == 0:
        #     import ipdb; ipdb.set_trace()

        # if bin_idxs[2] > 10 and bin_idxs[3] > 10:
        #     import ipdb; ipdb.set_trace()

        if i % 100 == 0:
            print(i)
            
        for pair in pairs:
            bin1 = bin_idxs[pair[0]]
            bin2 = bin_idxs[pair[1]]
            binned_similarities[pair][bin1][bin2].append(similarity)

            if (bin1, bin2) in to_plot[pair]:
                to_plot[pair][(bin1, bin2)] = (truth_v, predicted_v)

        # if i >= 500:
        #     break
                

    for pair in pairs:
        bins = binned_similarities[pair]

        avg_similarity = np.zeros(shape=(nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                avg_similarity[i, j] = np.mean(bins[i][j])

        paramname1 = model_cls.PARAM_NAMES[pair[0]]
        paramname2 = model_cls.PARAM_NAMES[pair[1]]

        range1 = model_cls.PARAM_RANGES[pair[0]]
        range2 = model_cls.PARAM_RANGES[pair[1]]

        plt.clf()
        fig = plt.figure(figsize=(12, 4))

        gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.5)
        heatmap_ax = plt.subplot(gs[:3, :3])
        trace_axs = [plt.subplot(gs[i, 3:]) for i in range(3)]

        im = heatmap_ax.imshow(avg_similarity, cmap='hot', aspect='equal', extent=(-1, 1, -1, 1))

        heatmap_ax.set_xlabel('Parameter: {}'.format(paramname1))
        heatmap_ax.set_ylabel('Parameter: {}'.format(paramname2))

        plt.gcf().colorbar(im, ax=heatmap_ax)

        # Plot traces
        colors = (('k', 'grey',), ('green', 'lime'), ('blue', 'cyan'))
        for i, ((bin1, bin2), (truth_v, predicted_v)) in enumerate(to_plot[pair].items()):
            # trace_axs[i].get_xaxis().set_visible(False)
            if i != len(trace_axs)-1:
                trace_axs[i].set_xticklabels([])
            if truth_v is None:
                continue
            # heatmap_ax.scatter([bin1], [bin2], c=colors[i], marker='o')
            x = 2 * (bin1-(nbins/2)) / nbins
            y = 2 * (bin2-(nbins/2)) / nbins
            dot = Circle((x, y), radius=.05, color=colors[i][1])
            heatmap_ax.add_patch(dot)

            # Plot traces
            x_axis = np.arange(0, len(predicted_v)*.02, .02)
            # t_range = slice(5000, 14000)
            t_range = slice(None)
            trace_axs[i].plot(
                x_axis[t_range], truth_v[t_range], color=colors[i][1], label='Actual',
                linewidth=0.5,
            )
            trace_axs[i].plot(
                x_axis[t_range], predicted_v[t_range], color=colors[i][0], label='Predicted',
                linewidth=0.5,
            )
            trace_axs[i].legend(bbox_to_anchor=(0.95, 0.8), loc=2)

        # trace_axs[-1].get_xaxis().set_visible(True)
        trace_axs[-1].set_xlabel("Time (ms)")
        trace_axs[-1].set_ylabel("Volts")
        
        if '--save' in sys.argv:
            plt.savefig('similarity/{}_avg_similarity_{}_vs_{}.png'.format(modelname, paramname1, paramname2))
        plt.show()



if __name__ == '__main__':
    similarity_heatmaps('izhi', 'cellRegr.sim.pred.h5') # 'izhi_v5b_chirp_16a_blind1.answer')

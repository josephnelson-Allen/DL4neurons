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

from models import MODELS_BY_NAME
from compute_similarity import Similarity


def histogram(exp_exp_similarity, exp_pred_similarity, sim_pred_similarity):
    pass


def similarity_heatmap(similarity_file, param_x=0, param_y=1, exp_exp_similarity=None, nbins=20, phys=True):
    """
    Compute heatmaps of similarity between actual/predicted traces.
    if exp_exp_similarity is specified, we normalize to its mean/stdev
    if phys=False, plot w/ unit-normalized params on axes
    """
    with h5py.File(similarity_file, 'r') as infile:
        modelname = infile['similarity'].attrs['modelname']
        param_ranges = MODELS_BY_NAME[modelname].PARAM_RANGES
        param_names = MODELS_BY_NAME[modelname].PARAM_NAMES
        nsamples = infile['similarity'].shape[0]

        if exp_exp_similarity:
            with h5py.File(exp_exp_similarity, 'r') as infile:
                expexp_mean = np.average(infile['similarity'])
                expexp_std = np.stdev(infile['similarity'])
                # TODO: normalize incoming data

        if phys:
            range_x = param_ranges[param_x]
            range_y = param_ranges[param_y]
            paramskey = 'physTruth2D'
            pred_paramskey = 'physPred2D'
        else:
            range_x = (-1, 1)
            range_y = (-1, 1)
            paramskey = 'unitTruth2D'
            pred_paramskey = 'unitPred2D'

        all_bins_x = np.linspace(*range_x, nbins)
        all_bins_y = np.linspace(*range_y, nbins)

        bins_x = np.digitize(infile[paramskey][:, param_x], all_bins_x)
        bins_y = np.digitize(infile[paramskey][:, param_y], all_bins_y)

        binned_similarities = defaultdict(list)
        for bin_x, bin_y, similarity in zip(bins_x, bins_y, infile['similarity']):
            binned_similarities[(bin_x, bin_y)].append(similarity)

        binned_averaged_similarity = np.zeros(shape=(nbins, nbins))
        for (bin_x, bin_y), similarities in binned_similarities.items():
            binned_averaged_similarity[bin_x, bin_y] = np.average(similarities)

        # Grab 3 random sets of params to plot traces for
        trace_i = sorted(np.random.randint(0, nsamples, size=3))
        true_params = infile[paramskey][trace_i, :]
        pred_params = infile[pred_paramskey][trace_i, :]

            
    # Display with correct axis labeling
    plt.clf()
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.5)
    heatmap_ax = plt.subplot(gs[:3, :3])
    trace_axs = [plt.subplot(gs[i, 3:]) for i in range(3)]

    cmap = 'hot' # TODO: Use one w/o white, use custom
    
    im = heatmap_ax.pcolor(all_bins_x, all_bins_y, binned_averaged_similarity, cmap=cmap)
    plt.colorbar(im, ax=heatmap_ax)
    heatmap_ax.set_xlabel('Parameter: {}'.format(param_names[param_x]))
    heatmap_ax.set_ylabel('Parameter: {}'.format(param_names[param_y]))

    # Plot traces
    x_axis = np.arange(0, .02*9000, .02)
    colors = (('k', 'grey',), ('green', 'lime'), ('blue', 'cyan'))
    for true_par, pred_par, ax, (col_pred, col_true) in zip(true_params, pred_params, trace_axs, colors):
        sim = Similarity(modelname, 'stims/chirp23a.csv')
        true_v = sim._data_for(*true_par, unit=not phys) 
        pred_v = sim._data_for(*pred_par, unit=not phys)
        trace_similarity = sim._similarity(true_v, pred_v)
        ax.plot(x_axis, pred_v, linewidth=0.5, label='Predicted', color=col_pred)
        ax.plot(x_axis, true_v, linewidth=0.5, label='Actual', color=col_true)
        ax.text(1.02, 0.8, "sim. = {0:.2f}".format(trace_similarity), transform=ax.transAxes)
        ax.legend(bbox_to_anchor=(0.95, 0.8), loc=2)

        dot = Circle((true_par[param_x], true_par[param_y]), radius=0.04, color=col_true)
        heatmap_ax.add_patch(dot)

    trace_axs[0].set_xticklabels([])
    trace_axs[1].set_xticklabels([])
    trace_axs[-1].set_xlabel("Time (ms)")
    trace_axs[-1].set_ylabel("Volts")

    if '--save' in sys.argv:
        plt.savefig('similarity/{}_avg_similarity_{}_vs_{}.png'.format(modelname, param_names[param_x], param_names[param_y]))
    plt.show()


if __name__ == '__main__':
    # similarity_heatmaps('izhi', 'cellRegr.sim.pred.h5') # 'izhi_v5b_chirp_16a_blind1.answer')
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for x, y, in pairs:
        similarity_heatmap('cellRegr.sim.pred_SimPredSimilarity.h5', param_x=x, param_y=y, phys=False)

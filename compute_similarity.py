"""
Compute all-pairs similarity within an experimental block and save to disk
"""
from argparse import ArgumentParser
import logging as log
import itertools

import numpy as np
import h5py
import pyspike

from models import MODELS_BY_NAME

ML_MODEL_i = 1 # Of 32 trained models, which one's predictions to use

# RIGHT NOW, ALL VOLTAGE DATA IS KEPT AS 9000 TIMEBIN TRACES


class Similarity(object):
    def __init__(self, modelname, stimfile, *args, **kwargs):
        self.model_cls = MODELS_BY_NAME[modelname]
        self.stim = np.genfromtxt(stimfile) * self.model_cls.STIM_MULTIPLIER

    def _data_for(self, *phys_params, dt=0.02, celsius=37, stim=None):
        stim = stim or self.stim
        return self.model_cls(*phys_params, log=log, celsius=celsius).simulate(stim, dt=dt)['v'][5500:14500]

    def _similarity(self, v1, v2, method='isi', **kwargs):
        if method == 'isi':
            thresh = kwargs.pop('thresh', 10)
            
            spikes1 = np.diff( (v1 > thresh).astype('int') )
            spikes2 = np.diff( (v2 > thresh).astype('int') )

            spike_times_1 = np.where(spikes1 > 0)[0]
            spike_times_2 = np.where(spikes2 > 0)[0]

            # spike_train_1 = pyspike.SpikeTrain(spike_times_1, 5500, 14500)
            # spike_train_2 = pyspike.SpikeTrain(spike_times_2, 5500, 14500)

            spike_train_1 = pyspike.SpikeTrain(spike_times_1, 0, 9000)
            spike_train_2 = pyspike.SpikeTrain(spike_times_2, 0, 9000)

            return np.abs(pyspike.isi_distance(spike_train_1, spike_train_2))
        else:
            raise ValueError("unknown similarity metric")

    def iter_exp_block(self, sweepfile, block_start, block_end):
        """
        For each trial, yield the experimental trace and predicted params.
        """
        with h5py.File(sweepfile, 'r') as infile:
            if block_end == -1:
                block_end = infile['sweep2D'].shape[0]
            
            # TODO: read whole block at once?
            for i in range(block_start, block_end):
                v_exp = infile['sweep2D'][i, :]
                phys_pred = infile['physPred3D'][i, :, ML_MODEL_i]
                yield v_exp, phys_pred

    def iter_sim_predictions(self, predfile):
        """
        For each simulated sample, yield the true params, predicted params, true trace
        """
        pass # TODO

    def iter_exp_pred_similarity(self, sweepfile, block_start, block_end):
        """
        Iterate over experimental sweeps: voltage trace and predicted params
        Re-run the simulation w/ predicted params
        Compare the experimental and simulated traces, and yield the similarities
        """
        log.info('Starting exp/pred similarity computation')
        for v_exp, phys_pred in self.iter_exp_block(sweepfile, block_start, block_end):
            v_sim = self._data_for(*phys_pred)
            yield self._similarity(v_exp, v_sim)
    
    def save_exp_pred_similarity(self, sweepfile, outfilename, block_start, block_end):
        with h5py.File(outfilename, 'w') as outfile:
            data = np.array(list(self.iter_exp_pred_similarity(sweepfile, block_start, block_end)))
            outfile.create_dataset('similarity', data=data)

    def iter_exp_exp_similarity(self, sweepfile, block_start, block_end):
        """
        Iterate over pairs of experimental sweeps: voltage traces only
        Yield exp/exp similarities
        """
        log.info('Starting exp/exp similarity computation')
        for (v1, _), (v2, _) in itertools.combinations(self.iter_exp_block(sweepfile, block_start, block_end), 2):
            yield self._similarity(v1, v2)

    def save_exp_exp_similarity(self, sweepfile, outfilename, block_start, block_end):
        with h5py.File(outfilename, 'w') as outfile:
            data = np.array(list(self.iter_exp_exp_similarity(sweepfile, block_start, block_end)))
            outfile.create_dataset('similarity', data=data)

    def iter_sim_pred_similarity(self, predfile):
        """
        Iterate over predictions file: true volts, true params, predicted params
        Compute the trace from predicted params
        Yield similarity between trace from predicted vs true params
        """
        log.info('Starting sim/pred similarity computation')

            
def main(args):
    x = Similarity(args.model, 'stims/chirp23a.csv')
    
    exp_pred_outfile = args.sweepfile.replace('.h5', '_ExpPredSimilarity.h5')
    x.save_exp_pred_similarity(args.sweepfile, exp_pred_outfile, args.block_start, args.block_end)
    
    exp_exp_outfile = args.sweepfile.replace('.h5', '_ExpExpSimilarity.h5')
    x.save_exp_exp_similarity(args.sweepfile, exp_exp_outfile, args.block_start, args.block_end)

    
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model', choices=MODELS_BY_NAME.keys(), default='izhi')
    parser.add_argument('--sweepfile', type=str, required=True)
    # parser.add_argument('--outfile', type=str, required=False, default=None)
    parser.add_argument('--block-start', type=int, default=0)
    parser.add_argument('--block-end', type=int, default=-1)
    
    parser.add_argument('--dist', choices=['isi'], default='isi')

    args = parser.parse_args()

    main(args)

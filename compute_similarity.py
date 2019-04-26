"""
Compute all-pairs similarity within an experimental block and save to disk
"""
from argparse import ArgumentParser
import logging
import itertools

import numpy as np
import h5py
import pyspike

from models import MODELS_BY_NAME

ML_MODEL_i = 1 # Of 32 trained models, which one's predictions to use

# RIGHT NOW, ALL VOLTAGE DATA IS KEPT AS 9000 TIMEBIN TRACES

log = logging.getLogger('calc_ecp')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
log.addHandler(ch)


class Similarity(object):
    def __init__(self, modelname, stimfile, *args, **kwargs):
        self.modelname = modelname
        self.model_cls = MODELS_BY_NAME[modelname]
        self.stim = np.genfromtxt(stimfile) * self.model_cls.STIM_MULTIPLIER

    def _rangeify(self, params):
        _r = lambda data, _range: (data + 1) * (_range[1] - _range[0])/2.0 + _range[0]
        return [_r(x, _range) for x, _range in zip(params, self.model_cls.PARAM_RANGES)]
        
    def _data_for(self, *params, dt=0.02, celsius=37, stim=None, unit=False):
        """
        If unit=True, rangeify first
        """
        stim = stim or self.stim
        phys_params = self._rangeify(params) if unit else params
        return self.model_cls(*phys_params, log=log, celsius=celsius).simulate(stim, dt=dt)['v'][5500:14500]

    def _similarity(self, v1, v2, method='isi', **kwargs):
        assert len(v1) == len(v2)
        if method == 'isi':
            thresh = kwargs.pop('thresh', 10)
            
            spikes1 = np.diff( (v1 > thresh).astype('int') )
            spikes2 = np.diff( (v2 > thresh).astype('int') )

            spike_times_1 = np.where(spikes1 > 0)[0]
            spike_times_2 = np.where(spikes2 > 0)[0]

            # spike_train_1 = pyspike.SpikeTrain(spike_times_1, 5500, 14500)
            # spike_train_2 = pyspike.SpikeTrain(spike_times_2, 5500, 14500)

            spike_train_1 = pyspike.SpikeTrain(spike_times_1, 9000*.02)
            spike_train_2 = pyspike.SpikeTrain(spike_times_2, 9000*.02)

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

    def iter_sim_predictions(self, predfile, print_every=100):
        """
        For each simulated sample, yield the predicted params(phys), predicted params(unit),
        true params(unit), true trace
        """
        with h5py.File(predfile, 'r') as infile:
            for i, (phys_pred, unit_pred, unit_truth, trace_truth) in enumerate(zip(infile['physPred2D'], infile['unitPred2D'], infile['unitTruth2D'], infile['trace2D'])):
                if print_every and i%print_every == 0:
                    log.info('Done {}'.format(i))
                if len(trace_truth) > 9000:
                    trace_truth = trace_truth[5500:14500]
                yield phys_pred, unit_pred, unit_truth, trace_truth

                # DEBUG
                # if i > 500:
                #     break
                # END DEBUG

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

            with h5py.File(sweepfile, 'r') as infile:
                outfile.create_dataset('physPred3D', data=infile['physPred3D'][:, :, ML_MODEL_i])
                outfile.create_dataset('unitPred3D', data=infile['unitPred3D'][:, :, ML_MODEL_i])

    def iter_sim_pred_similarity(self, predfile):
        """
        Iterate over predictions file: true volts, true params, predicted params
        Compute the trace from predicted params
        Yield similarity between trace from predicted vs true params
        """
        log.info('Starting sim/pred similarity computation')
        
        for phys_pred, unit_pred, unit_truth, v_truth in self.iter_sim_predictions(predfile):
            # phys_truth = self._rangeify(unit_truth)
            v_pred = self._data_for(*phys_pred)
            yield self._similarity(v_truth, v_pred)

    def save_sim_pred_similarity(self, predfile, outfilename):
        # TODO: save params too, direct from other h5 file
        with h5py.File(outfilename, 'w') as outfile:
            data = np.array(list(self.iter_sim_pred_similarity(predfile)))
            outfile.create_dataset('similarity', data=data)
            outfile['similarity'].attrs['modelname'] = self.modelname

            with h5py.File(predfile, 'r') as infile:
                outfile.create_dataset('physPred2D', data=infile['physPred2D'])
                outfile.create_dataset('unitPred2D', data=infile['unitPred2D'])
                outfile.create_dataset('unitTruth2D', data=infile['unitTruth2D'])

            
def main(args):
    x = Similarity(args.model, 'stims/chirp23a.csv')
    
    exp_pred_outfile = args.sweepfile.replace('.h5', '_ExpPredSimilarity.h5')
    x.save_exp_pred_similarity(args.sweepfile, exp_pred_outfile, args.block_start, args.block_end)
    
    exp_exp_outfile = args.sweepfile.replace('.h5', '_ExpExpSimilarity.h5')
    x.save_exp_exp_similarity(args.sweepfile, exp_exp_outfile, args.block_start, args.block_end)

    sim_pred_outfile = args.simpredfile.replace('.h5', '_SimPredSimilarity.h5')
    x.save_sim_pred_similarity(args.simpredfile, sim_pred_outfile)

    
if __name__ == '__main__':
    parser = ArgumentParser()

    # VB: Try running with --sweepfile 041019A_1-ML203b_izhi_4pv6c.mpred.h5 --block-start 90 --block-end 150 --simpredfile cellRegr.sim.pred.h5 


    parser.add_argument('--model', choices=MODELS_BY_NAME.keys(), default='izhi')
    parser.add_argument('--sweepfile', type=str, required=True)
    parser.add_argument('--simpredfile', type=str, required=True)
    # parser.add_argument('--outfile', type=str, required=False, default=None)
    parser.add_argument('--block-start', type=int, default=0)
    parser.add_argument('--block-end', type=int, default=-1)
    
    parser.add_argument('--dist', choices=['isi'], default='isi')

    args = parser.parse_args()

    main(args)
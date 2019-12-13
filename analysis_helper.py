# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:59:43 2019

@author: bensr
"""

from run import get_model
import logging as log
import models
import numpy as np
import os
import matplotlib.pyplot as plt
def make_paramset(my_model,unpredicted_params_ind,nsamples):
    def_param_vals = my_model.DEFAULT_PARAMS
    param_set = np.array([def_param_vals]*nsamples)
    range_to_vary = my_model.PARAM_RANGES[unpredicted_params_ind[0]]
    vals_check = np.linspace(range_to_vary[0],range_to_vary[1],nsamples)
    param_set[:,unpredicted_params_ind[0]] = vals_check
    return param_set
    
    
    
def get_stim(stimfn):
    return np.genfromtxt(stimfn, dtype=np.float32)
stim_fn = './stims/chaotic_1.csv'
stim = get_stim(stim_fn)
#scan a paramter range
def get_volts(nsamples):
    m_type = 'L1_DAC'
    e_type ='bNAC'
    all_volts = []
    unpredicted_params_ind=[0]
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0) 
    param_set = make_paramset(my_model,unpredicted_params_ind,nsamples)
    for i in range(nsamples):
        my_model = get_model('BBP',log,m_type,e_type,0,*param_set[i]) 
        volts = my_model.simulate(stim,0.02)
        all_volts.append(volts)
    return volts
def analyze_volts(all_volts):
    times = [0.02*i for i in range(len(stim))]
    cum_sum_volts = []
    for curr_volts in all_volts:
        probes = list(curr_volts.keys())
        res = [i for i in probes if 'axon' in i] 
        volts_to_plot = curr_volts.get(res[0])
        cum_sum_volts.append(np.cumsum(volts_to_plot))
        plt.plot(times,volts_to_plot[:-1])
    plt.show()
all_volts = get_volts(5)
analyze_volts(all_volts)
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

def get_stim(stimfn):
    return np.genfromtxt(stimfn, dtype=np.float32)
stim_fn = './stims/chaotic_1.csv'
stim = get_stim(stim_fn)
#scan a paramter range
def get_volts(nsamples):
    m_type = 'L1_DAC'
    e_type ='bNAC'
    all_volts = []
    
    
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0) 
    unpredicted_params_ind = [2]
    def_param_vals = my_model.DEFAULT_PARAMS
    param_names =  my_model.PARAM_NAMES
    for pname,pval in zip(param_names,def_param_vals):
        setattr(my_model, pname, pval)
    range_to_vary = my_model.PARAM_RANGES[unpredicted_params_ind[0]]
    vals_check = np.random.uniform(range_to_vary[0],range_to_vary[1],nsamples)
    for pval in vals_check:
        #my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0) 
        setattr(my_model, param_names[unpredicted_params_ind[0]], pval)
        volts = my_model.simulate(stim,0.02)
        all_volts.append(volts)
    return all_volts
def analyze_volts(all_volts):
    times = [0.02*i for i in range(len(stim))]
    for curr_volts in all_volts:
        volts_to_plot = curr_volts.get('bNAC219_L1_DAC_0d58fdf14a[1].axon[1]')
        plt.plot(times,volts_to_plot[:-1])
    plt.show()
all_volts = get_volts(3)
analyze_volts(all_volts)
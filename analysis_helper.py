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
import json
import itertools


stimfn = './stims/chaotic_1.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32) 
plt.subplots_adjust(hspace=0.3)
times = [0.02*i for i in range(len(stim))]


def make_paramset(my_model,param_ind,nsamples):
    def_param_vals = my_model.DEFAULT_PARAMS
    param_set = np.array([def_param_vals]*nsamples)
    range_to_vary = my_model.PARAM_RANGES[param_ind]
    vals_check = np.linspace(range_to_vary[0],range_to_vary[1],nsamples)
    param_set[:,param_ind] = vals_check
    return param_set 

def get_volts(mtype,etype,param_ind,nsamples):
    all_volts = []
    my_model = get_model('BBP',log,m_type=mtype,e_type=etype,cell_i=0) 
    param_set = make_paramset(my_model,param_ind,nsamples)
    for i in range(nsamples):
        my_model = get_model('BBP',log,mtype,etype,0,*param_set[i]) 
        volts = my_model.simulate(stim,0.02)
        all_volts.append(volts)
    return all_volts
def get_rec_sec(def_volts,adjusted_param):
    probes = list(def_volts.keys())
    if 'soma' in adjusted_param:
        rec_sec = probes[0]
    if 'dend' in adjusted_param:
        res = [i for i in probes if 'dend' in i]
        rec_sec = res[2]   
    if 'axon' in adjusted_param:
        res = [i for i in probes if 'axon' in i]
        rec_sec = res[2]  
    return rec_sec
    
def check_param_sensitivity(all_volts,def_volts_probes,adjusted_param):
    fig, (ax1,ax2,ax3)= plt.subplots(3)
    fig.suptitle(adjusted_param)
    rec_sec = get_rec_sec(def_volts_probes,adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    def_volts = def_volts_probes.get(rec_sec)
    ax1.plot(times,def_volts[:-1],'black')
    def_cum_sum = np.cumsum(np.abs(def_volts))*0.02
    cum_sum_errs = []
    plt.subplots_adjust(hspace=0.3)
    for curr_volts in all_volts:
        volts_to_plot = curr_volts.get(rec_sec)
        curr_cum_sum= np.cumsum(np.abs(volts_to_plot))*0.02
        cum_sum_err = curr_cum_sum - def_cum_sum
        err = def_volts - volts_to_plot
        ax1.plot(times,volts_to_plot[:-1])
        ax2.plot(times,err[:-1])
        ax3.plot(times,cum_sum_err[:-1])
        cum_sum_errs.append(cum_sum_err)
    ax1.title.set_text('Volts')
    ax2.title.set_text('error')
    ax3.title.set_text('cum_sum_error')
    fig_name = m_type + e_type + adjusted_param +'.pdf'
    fig.savefig(fig_name)
    return cum_sum_errs
#analyze_volts([])
#with open('cells.json') as infile:
#        cells = json.load(infile)
#        ALL_MTYPES = cells.keys()
#        ALL_ETYPES = list(set(itertools.chain.from_iterable(mtype.keys() for mtype in cells.values())))

def test_sensitivity(mtype,etype):
    my_model = get_model('BBP',log,m_type=mtype,e_type=etype,cell_i=0) 
    def_volts = []#my_model.simulate(stim,0.02)
    param_names = my_model.PARAM_NAMES
    for i in range(len(param_names)):
        adjusted_param = my_model.PARAM_NAMES[i]
        all_volts = get_volts(mtype,etype,i,2)
        curr_errs = check_param_sensitivity(all_volts,def_volts,adjusted_param)
        curr_ECDs = cur_errs[:,-1]
        
m_type = 'L1_DAC'
e_type ='bNAC'
test_sensitivity(m_type,e_type)
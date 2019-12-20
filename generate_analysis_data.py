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
import pickle as pkl
import random
stimfn = './stims/chaotic_1.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32) 
plt.subplots_adjust(hspace=0.3)
times = [0.02*i for i in range(len(stim))]
m_type = 'L1_DAC'
e_type ='bNAC'

def make_paramset(my_model,param_ind,nsamples):
    def_param_vals = my_model.DEFAULT_PARAMS
    param_set = np.array([def_param_vals]*nsamples)
    range_to_vary = my_model.PARAM_RANGES[param_ind]
    #vals_check = np.linspace(range_to_vary[0],range_to_vary[1],nsamples)
    vals_check=def_param_vals[param_ind]*np.exp(np.random.uniform(-1,1,size=nsamples)*np.log(10))
    param_set[:,param_ind] = vals_check
    return param_set 

def get_volts(mtype,etype,param_ind,nsamples):
    all_volts = []
    my_model = get_model('BBP',log,m_type=mtype,e_type=etype,cell_i=1) 
    param_set = make_paramset(my_model,param_ind,nsamples)
    #param_name = my_model.PARAM_NAMES[param_ind]
    for i in range(nsamples):
        print("working on param_ind" + int(param_ind) + " sample" + int(i))
        params = param_set[i]
        my_model = get_model('BBP',log,mtype,etype,1,*params)
        my_model.DEFAULT_PARAMS = False
        volts = my_model.simulate(stim,0.02)
        all_volts.append(volts)
    return all_volts
def get_rec_sec(def_volts,adjusted_param):
    probes = list(def_volts.keys())
    rec_sec=adjusted_param
    if 'soma' in adjusted_param:
        rec_sec = probes[0]
    if 'apic' in adjusted_param or 'dend' in adjusted_param:
        res = [i for i in probes if 'apic' in i or 'dend' in i]
        rec_sec = res[2]   
    if 'axon' in adjusted_param:
        res = [i for i in probes if 'axon' in i]
        rec_sec = res[2]  
    dot_ind = rec_sec.find('.')+1
    return rec_sec[dot_ind:],rec_sec[:dot_ind]
    
def check_param_sensitivity(all_volts,def_volts_probes,adjusted_param):
    fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
    fig.suptitle(adjusted_param)
    def_rec_sec,prefix = get_rec_sec(def_volts_probes,adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    def_volts = def_volts_probes.get(prefix + def_rec_sec)
    ax1.plot(times,def_volts[:-1],'black')
    def_cum_sum = np.cumsum(np.abs(def_volts))*0.02
    cum_sum_errs = []
    plt.subplots_adjust(hspace=0.3)
    for curr_volts in all_volts:
        curr_rec_sec,prefix = get_rec_sec(curr_volts,adjusted_param)
        if (curr_rec_sec != def_rec_sec):
            print("curr_rec_sec is " + curr_rec_sec + 'and def rec_sec is' + def_rec_sec )
        volts_to_plot = curr_volts.get(prefix +def_rec_sec)
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

def main():
    nsamples = 2
    try:
        procid = int(os.environ['SLURM_PROCID'])
        print("in cori")
        nsamples = 2
    except:
        print("not in cori")
        procid = 0
        nsamples = 2        
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=1) 
    adjusted_param = my_model.PARAM_NAMES[procid]
    print("working on " + adjusted_param)
    all_volts = get_volts(m_type,e_type,procid,nsamples)
    pkl_fn=m_type + e_type + adjusted_param +'.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_volts,output)
main()

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
import time
import sys
stimfn = './stims/chaotic_1.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32)
plt.subplots_adjust(hspace=0.3)
times = [0.02*i for i in range(len(stim))]
def make_paramset(my_model,param_ind,nsamples):
    def_param_vals = my_model.DEFAULT_PARAMS
    param_set = np.array([def_param_vals]*nsamples)
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
        s_time = time.time()
        print("working on param_ind" + str(param_ind) + " sample" + str(i))
        params = param_set[i]
        my_model = get_model('BBP',log,mtype,etype,1,*params)
        my_model.DEFAULT_PARAMS = False
        volts = my_model.simulate(stim,0.02)
        all_volts.append(volts)
        end_time = time.time()
        etime = end_time-s_time
        print("done param_ind" + str(param_ind) + " sample" + str(i)+str(etime))
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
m_type = 'L1_DAC'
e_type ='bNAC'
def main():
    NTHREADS = 128
    m_type = sys.argv[1]
    e_type = sys.argv[2]
    nsamples = int(sys.argv[3])
    
    try:
        procid = int(os.environ['SLURM_PROCID'])
        print("in cori")
        
    except:
        print("not in cori")
        procid = 0   
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=1)
    
    def_vals = my_model.DEFAULT_PARAMS
    pnames = [my_model.PARAM_NAMES[i] for i in range(len(def_vals)) if def_vals[i]>0]
    threads_per_param = int(NTHREADS/len(pnames))
    samples_per_thread = int(nsamples/threads_per_param)+1
    p_ind = procid%(len(pnames))
    adjusted_param = my_model.PARAM_NAMES[p_ind]
    print("working on " + adjusted_param)
    all_volts = get_volts(m_type,e_type,p_ind,samples_per_thread)
    pkl_fn=m_type + e_type + adjusted_param + '_' + procid + '.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_volts,output)
main()
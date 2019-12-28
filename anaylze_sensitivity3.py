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
import sys
stimfn = './stims/chaotic_1.csv'
stim =  np.genfromtxt(stimfn, dtype=np.float32) 
plt.subplots_adjust(hspace=0.3)
times = [0.02*i for i in range(len(stim))]
create_pdfs = True

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

  
def check_param_sensitivity(all_volts,adjusted_param,files_loc):
    print(adjusted_param)
    def_rec_sec,prefix = get_rec_sec(all_volts[0],adjusted_param)
     #in probe the first will always be the soma then axon[0] (AIS) then a sec that has mid (0.5) distrance
    #ax1.plot(times,def_volts[:-1],'black')
    volt_debug = []
    cum_sum_errs = []
    if(create_pdfs):
        fig, (ax1,ax2,ax3)= plt.subplots(3,figsize=(15,15))
        fig.suptitle(adjusted_param)
        plt.subplots_adjust(hspace=0.3)
        
    for i in range(int(len(all_volts)/2)):
        
        volts1 = all_volts[2*i]
        volts2 = all_volts[2*i + 1]
        
        curr_rec_sec1,prefix1 = get_rec_sec(volts1,adjusted_param)
        curr_rec_sec2,prefix2 = get_rec_sec(volts2,adjusted_param)
        if (curr_rec_sec1 != curr_rec_sec2):
            print("curr_rec_sec is " + curr_rec_sec1 + 'and curr_rec_sec2  is' + curr_rec_sec2 )
        volts_to_plot1 = volts1.get(prefix1 +def_rec_sec)
        volts_to_plot2 = volts2.get(prefix2 +def_rec_sec)
        volt_debug.append(volts_to_plot1)
        volt_debug.append(volts_to_plot2)
        curr_cum_sum1= np.cumsum(np.abs(volts_to_plot1))*0.02
        curr_cum_sum2= np.cumsum(np.abs(volts_to_plot2))*0.02
        cum_sum_err = curr_cum_sum1 - curr_cum_sum2
        cum_sum_errs.append(cum_sum_err)
        if(create_pdfs):
            err = volts_to_plot1 - volts_to_plot2
            ax1.plot(times,volts_to_plot1[:-1])
            ax1.plot(times,volts_to_plot2[:-1])
            ax2.plot(times,err[:-1])
            ax3.plot(times,cum_sum_err[:-1])
    if(create_pdfs):   
        ax1.title.set_text('Volts')
        ax2.title.set_text('error')
        ax3.title.set_text('cum_sum_error')
        fig_name = adjusted_param +'.pdf'
        fig.savefig(files_loc + fig_name)
    volt_debug = np.array(volt_debug)
    return cum_sum_errs



def test_sensitivity(files_loc,my_model):
    param_names = my_model.PARAM_NAMES
    all_ECDS ={}
    all_fns = os.listdir(files_loc)
    for i in range(len(param_names)):
        adjusted_param = param_names[i]
        param_files = [files_loc + fn for fn in all_fns if adjusted_param in fn]
        param_files = [ fn for fn in param_files if '.pkl' in fn]
        all_volts = []
        for fn in param_files:
            with open(fn, 'rb') as f:
                curr_volts = pkl.load(f)
                all_volts = all_volts + curr_volts
        if len(all_volts)>0:
            curr_errs = check_param_sensitivity(all_volts,adjusted_param,files_loc)
            curr_ECDs = [ls[-1] for ls in curr_errs]
            all_ECDS[adjusted_param]=curr_ECDs
    pkl_fn=files_loc + my_model.m_type + my_model.e_type + 'sensitivity.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(all_ECDS,output)
        pkl.dump(param_names,output)
    return all_ECDS


def analyze_ecds(ECDS,def_vals,files_loc):
    ymx_axon = 1000
    ymx_dend = 300
    ymx_soma = 600
    threshold_axon = 100
    threshold_soma = 100
    threshold_dend = 100
    param_names = list(ECDS.keys())
    pnames_soma = []
    pnames_axon = []
    pnames_dend = []
    means_axon = []
    STDs_axon = []
    means_soma = []
    STDs_soma = []
    means_dend = []
    STDs_dend = []
    params_sensitivity_dict = {}
    param_inds = range(len(param_names))
    for i in param_inds:
        if def_vals[i] <= 0:
            continue
        curr_ecds = ECDS[param_names[i]]
        nsamples = len(curr_ecds)
        curr_mean = np.mean(curr_ecds)
        curr_std = np.std(curr_ecds)
        params_sensitivity_dict[param_names[i]] = [curr_mean,curr_std]
        if 'soma' in param_names[i]:
            pnames_soma.append(param_names[i][0:5])
            means_soma.append(curr_mean)
            STDs_soma.append(curr_std)
        if 'apic' in param_names[i] or 'dend' in param_names[i]:
            pnames_dend.append(param_names[i][0:5])
            means_dend.append(curr_mean)
            STDs_dend.append(curr_std)
        if 'axon' in param_names[i]:
            pnames_axon.append(param_names[i][0:5])
            means_axon.append(curr_mean)
            STDs_axon.append(curr_std)
          
    pkl_fn=files_loc + 'mean_std_sensitivity'  + str(nsamples) + '.pkl'
    with open(pkl_fn, 'wb') as output:
        pkl.dump(params_sensitivity_dict,output)
    fig, ((ax_soma,ax_dend),( ax_axon,ax4))= plt.subplots(2,2,figsize=(15,15))
    fig.suptitle('Sensitivity analysis mean/rms ' + sys.argv[1] + sys.argv[2])
    
    ax_axon.title.set_text('Axonal del_ecds')
    means_axon = np.array(means_axon)
    STDs_axon = np.array(STDs_axon)
    yaxis_axon = np.divide(means_axon,STDs_axon)
    #yaxis_axon = np.clip(yaxis_axon,0,1)
    ax_axon.plot(range(len(pnames_axon)),yaxis_axon,'o')
    ax_axon.set_xticks(range(len(pnames_axon)))
    ax_axon.set_xticklabels(pnames_axon)
    ax_axon.grid()
    #ax_axon.set_ylim([0,1])
    ax_axon.set_ylabel('avr/std')
    
    
    ax_soma.title.set_text('somaal del_ecds')
    means_soma = np.array(means_soma)
    STDs_soma = np.array(STDs_soma)
    yaxis_soma = np.divide(means_soma,STDs_soma)
    #yaxis_soma = np.clip(yaxis_soma,0,1)
    ax_soma.plot(range(len(pnames_soma)),yaxis_soma,'o')
    ax_soma.set_xticks(range(len(pnames_soma)))
    ax_soma.set_xticklabels(pnames_soma)
    ax_soma.grid()
    #ax_soma.set_ylim([0,1])
    ax_soma.set_ylabel('avr/std')
    
    ax_dend.title.set_text('dendal del_ecds')
    means_dend = np.array(means_dend)
    STDs_dend = np.array(STDs_dend)
    yaxis_dend = np.divide(means_dend,STDs_dend)
    #yaxis_dend = np.clip(yaxis_dend,0,1)
    ax_dend.plot(range(len(pnames_dend)),yaxis_dend,'o')
    ax_dend.set_xticks(range(len(pnames_dend)))
    ax_dend.set_xticklabels(pnames_dend)
    ax_dend.grid()
    #ax_dend.set_ylim([0,1])
    ax_dend.set_ylabel('avr/std')
    
    
    fig_name = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sampling_size.pdf'
    fig.savefig(files_loc + fig_name)
    
   
    fig1, ((ax_soma1,ax_dend1),( ax_axon1,ax4))= plt.subplots(2,2,figsize=(15,15))
    fig1.suptitle('Sensitivity analysis RMS ' + sys.argv[1] + sys.argv[2])
    
    STDs_axon = np.clip(STDs_axon,0,ymx_axon)
    ax_axon1.title.set_text('Axonal Parameters RMS clipped at ' + str(ymx_axon))
    ax_axon1.plot(range(len(pnames_axon)),STDs_axon,'o')
    ax_axon1.set_xticks(range(len(pnames_axon)))
    ax_axon1.set_xticklabels(pnames_axon)
    ax_axon1.axhline(threshold_axon,color='red')
    ax_axon1.grid()
    
    STDs_soma = np.clip(STDs_soma,0,ymx_soma)
    ax_soma1.title.set_text('somaal Parameters RMS clipped at ' + str(ymx_soma))
    ax_soma1.plot(range(len(pnames_soma)),STDs_soma,'o')
    ax_soma1.set_xticks(range(len(pnames_soma)))
    ax_soma1.set_xticklabels(pnames_soma)
    ax_soma1.axhline(threshold_soma,color='red')
    ax_soma1.grid()
    
    STDs_dend = np.clip(STDs_dend,0,ymx_dend)
    ax_dend1.title.set_text('dendal Parameters RMS clipped at ' + str(ymx_dend))
    ax_dend1.plot(range(len(pnames_dend)),STDs_dend,'o')
    ax_dend1.set_xticks(range(len(pnames_dend)))
    ax_dend1.set_xticklabels(pnames_dend)
    ax_dend1.axhline(threshold_dend,color='red')
    ax_dend1.grid()
    
    fig_name1 = sys.argv[1] + sys.argv[2]  + str(nsamples) + 'Analysis_sensitivity_threshold.pdf'
    fig1.savefig(files_loc + fig_name1)
    #plt.show()
    
    return params_sensitivity_dict

def main():
    m_type = sys.argv[1]
    e_type = sys.argv[2]
    files_loc = './Sensitivity_output/' + m_type + '_' + e_type + '/'
    my_model = get_model('BBP',log,m_type=m_type,e_type=e_type,cell_i=0)
    def_vals = my_model.DEFAULT_PARAMS 
    ECDS = test_sensitivity(files_loc,my_model)
    analyze_ecds(ECDS,def_vals,files_loc)
    
main()
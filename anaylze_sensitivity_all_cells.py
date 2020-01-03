# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:59:55 2020

@author: bensr
"""
import urllib.request
import yaml
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle as pkl
import os
def get_ml_results():
    #data_loc = '/project/m2043/ML4neuron2b/' +short_name +'/cellSpike.sum_pred.yaml
    #data_loc = './Sensitivity_output/L1_DAC_cNAC/cellSpike.sum_pred.yaml'
    #bbp053/cellSpike.sum_pred.yaml
    
    folder_url = urllib.request.urlopen ( 'https://portal.nersc.gov/project/m2043/ML4neuron4b/')
    url_str = folder_url.read().decode('utf-8')
    tmp = url_str.split('href="bbp')
    folder_names = ['bbp'+ item[0:3] for item in tmp]
    folder_names = folder_names[1:]
    ml_quality = np.zeros((3,31))
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1,len(folder_names)+1)]
    ind = 0
    fig1, ax = plt.subplots(1,figsize=(15,15))
    result = {}
    for short_name in folder_names:
        ind +=1
        file_url = urllib.request.urlopen ( 'https://portal.nersc.gov/project/m2043/ML4neuron4b/' + short_name + '/cellSpike.sum_pred.yaml')
        ml_preds = yaml.safe_load(file_url)
        curr_preds = ml_preds['lossAudit']
        curr_STDs = [item[2] for item in curr_preds]
        pnames = [item[0]  for item in curr_preds]
        x_axis = range(len(curr_STDs))
        x_axis = [x + ind/(len(folder_names)) for x in x_axis]
        ax.plot(x_axis,curr_STDs,'o',color=colors[ind])
        for (std,i) in zip(curr_STDs,range(len(curr_STDs))):
            if not 'const' in pnames[i]:
                if std<0.4:
                    ml_quality[0,i] +=1
                elif std>0.5:
                    ml_quality[2,i] +=1  
                else:
                    ml_quality[1,i] +=1
        bbp_name = ml_preds['bbp_name']
        result[bbp_name] = [pnames,curr_STDs,ml_quality[:,ind]]
    pnames = [item[0]  for item in curr_preds]
    new_names = []
    csv_fn = 'ml_quality.csv'
    with open(csv_fn, 'w',newline='') as out_file:
        writer = csv.writer(out_file)
        for i in range(len(pnames)):
            curr_name = pnames[i]
            row_vals = list(ml_quality[:,i])
            row = [curr_name] + row_vals
            writer.writerow(row)
            new_name = curr_name + str(int(ml_quality[0,i])) + '/' + str(int(ml_quality[1,i])) + '/' + str(int(ml_quality[2,i])) 
            new_names.append(new_name)
        
    ax.set_xticks(range(len(new_names)))
    ax.set_xticklabels(new_names,rotation=90)
    ax.grid()
    ax.set_ylim([-0.1,0.6])
    fig1.savefig('ml_sens.pdf')
    plt.show()
    return result
    
def get_analysis_results():
    #ml_res = get_ml_results()
    #bbp_names = list(ml_res.keys())
    sens_summary = {}
    loc = './output/'
    dirs = os.listdir(loc)
    dirs = [x for x in dirs if '_' in x]
    for curr_dir in dirs:
        m1,m2,etype = curr_dir.split('_')
        mtype = m1 + '_' + m2
        if not 'cAD'  in etype:
            fn = loc + curr_dir + '/' + mtype + etype + 'mean_std_sensitivity.pkl'
            with open(fn, 'rb') as f:
                sensitivity_res = pkl.load(f)
            pnames = list(sensitivity_res.keys())
            for pname in pnames:
                curr_sens = sensitivity_res[pname]
                if pname not in sens_summary:
                    sens_summary[pname] = np.zeros(15)
                
                sens_std = curr_sens[1]
                if sens_std>140:
                    sens_ind = 14
                else:
                    sens_ind = int(sens_std/10)
                sens_summary[pname][sens_ind] +=1
    csv_fn = 'sens_summary.csv'
    with open(csv_fn, 'w',newline='') as out_file:
        writer = csv.writer(out_file)
        for pname in list(sens_summary.keys()):
            row_vals = list(sens_summary[pname])
            row = [pname] + row_vals
            writer.writerow(row)
    
        
            
                
        
get_analysis_results()     

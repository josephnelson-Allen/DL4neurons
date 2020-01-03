# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:59:55 2020

@author: bensr
"""
import numpy as np
import csv
import pickle as pkl
import os

    
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
            try:
                with open(fn, 'rb') as f:
                    sensitivity_res = pkl.load(f)
            except:
                print('cant find ' + fn)
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

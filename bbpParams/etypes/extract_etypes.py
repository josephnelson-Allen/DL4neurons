import csv
import numpy as np
import os
import shutil


readFrom =  "/global/cscratch1/sd/hdong/bbp/bbp_web_data/"
saveTo = './'
folders = ["L1/", "L23/", "L4/", "L5/", "L6/"]

count = 1
shortName = []
longName = []
types = []
variedParamsNum = []


for folder in folders:
    if '.' in folder:
        continue
    for subfolder in os.listdir(os.path.join(readFrom, folder)):
        if '.' in subfolder or "_1" not in subfolder:
            continue
        shortName.append("bbp" + ('0' * (4-len(str(count)))) + str(count))
        count += 1
        
        longName.append(subfolder)
        
        if 'cAD' in subfolder:
            types.append("cad")
        else:
            types.append("inh")
        
        variedParamsNum.append("NA")
        

zipped = list(zip(shortName, longName, types, variedParamsNum))

with open(os.path.join(saveTo, 'extracted_etypes.csv'), mode='w') as file:
    w = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(["Short Name", "Long Name", "Type", "Number of Varied Params"])
    w.writerows(zipped)
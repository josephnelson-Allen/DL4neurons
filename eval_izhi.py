"""
Load Jan's predictions, true values, and check closest izhi classification to each
"""
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np

from models import MODELS_BY_NAME

izhi = MODELS_BY_NAME['izhi']

def normalize(vals, minmax=1):
    mins = np.array([tup[0] for tup in izhi.PARAM_RANGES])
    ranges = np.array([_max - _min for (_min, _max) in izhi.PARAM_RANGES])
    return 2*minmax * ( (vals - mins)/ranges ) - minmax

# Normalized param values to classification name
classes = OrderedDict()

def load_classes():
    with open('modfiles/izhi2003a.mod', 'r') as modfile:
        while modfile.readline().strip() != 'COMMENT':
            pass

        headers = modfile.readline().split()
        modfile.readline() # space between headers and vals

        while True:
            line = modfile.readline().strip()
            if line == "ENDCOMMENT":
                break

            line = line.split()
            vals = normalize([float(x) for x in line[:4]])
            class_name = line[6].strip('% ')

            classes[tuple(vals)] = class_name

load_classes()

predictions = np.zeros((10000, 4), dtype=np.float64)
with open(sys.argv[1]) as prediction_file:
    for line in prediction_file.readlines():
        recid, vals = line.split('[')

        predictions[int(float(recid.strip()))] = vals.strip('] \n').split()

# true = np.genfromtxt(sys.argv[2])
true = np.zeros((10000, 4), dtype=np.float64)
with open(sys.argv[2], 'r') as true_param_file:
    for i, line in enumerate(true_param_file.readlines()):
        true[i] = normalize([float(x) for x in line.split()])


dist = lambda x, y: sum((a-b)**2 for a, b in zip(x, y))
for truth, prediction in zip(true, predictions):
    truth_distances = {dist(truth, class_params): class_name
                       for class_params, class_name in classes.items()}
    prediction_distances = {dist(prediction, class_params): class_name
                            for class_params, class_name in classes.items()}

    truth_class = truth_distances[min(truth_distances.keys())]
    predition_class = prediction_distances[min(prediction_distances.keys())]
    

    import ipdb; ipdb.set_trace()

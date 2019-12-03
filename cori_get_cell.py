"""
Script to generate the m-type or e-type from the slurm array task id
"""

import os, sys
import json

with open('cells.json') as infile:
    cells = json.load(infile)

all_m_types = sorted(cells.keys())

i = int(os.environ['SLURM_ARRAY_TASK_ID']) * 2

m_type = all_m_types[i]

e_type = sorted(cells[m_type].keys())[0]

if '--m-type' in sys.argv:
    print(m_type)

if '--e-type' in sys.argv:
    print(e_type)

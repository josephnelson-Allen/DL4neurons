import h5py
import matplotlib.pyplot as plt
import numpy as np
f = h5py.File('test.h5', 'r')
dset = f['voltages']
slice = dset[0,:,0]
time=np.arange(0,200,0.025)
plt.plot(time,slice)
plt.show()
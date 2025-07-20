#!/usr/bin/env python 

from glob import glob
import pickle 

f_dirs = glob('./qm_calc/frame_*/result.pkl')

data_all = []
for f in f_dirs:
    with open(f, 'rb') as f_in:
        data = pickle.load(f_in)
        data_all.append(data)

with open('data_all.pkl', 'wb') as f_out:
    pickle.dump(data_all, f_out)
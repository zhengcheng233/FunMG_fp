#!/usr/bin/env python 
import pickle 
import numpy as np
import os 

with open('mrtadf.pkl','rb') as fp:
    data = pickle.load(fp)

for ii in range(len(data)):
    os.makedirs(f'frame_{ii}',exist_ok=True)
    with open(f'frame_{ii}/result.pkl','wb') as fp:
        pickle.dump(data[ii],fp)




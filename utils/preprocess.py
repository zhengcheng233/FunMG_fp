#!/usr/bin/env python 
import numpy as np 
import pickle 
import os 

data = np.load('Ir_complex.npz',allow_pickle=True)
coords = data['coords']; symbols = data['symbols']
bonds = data['bonds']; conns = data['conns']

for ii in range(0, len(coords), 100):
    coord = coords[ii]; symbol = symbols[ii]
    bond = bonds[ii]; conn = conns[ii]
    os.makedirs(f'qmcalc/frame_{ii}', exist_ok=True)
    with open(f'qmcalc/frame_{ii}/result.pkl', 'wb') as f:
        result = {'coord_init': coord, 'symbol': symbol, 'bond': bond, 'conn': conn}
        pickle.dump(result, f)
    

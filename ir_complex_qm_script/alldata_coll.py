#!/usr/bin/env python
"""
the script used to collect data including plqy, e_ad, edme, homo, lumo
"""
import os
import json
from glob import glob
import numpy as np
from oled_opt_reason import topo_bond, mol_distance, ligand_check, bond_check2, bond_read, _get_atom
from input_gen import read_init_xyz
import sys
import math

#f_dirs = glob('./02.fp/conf.91_91_110')
f_dirs = glob('./iter.init/02.fp/conf.*')
f_dirs += glob('./iter.*/conf.*')

confs = []; coords = []; symbols = []
cwd = os.getcwd()
for dir0 in f_dirs:
    if os.path.isfile(os.path.join(dir0,'plqy.json')):
        confs.append(os.path.basename(dir0))
        os.chdir(dir0)
        element_xyz = read_init_xyz('t1-opt.log')
        coord = [u[1:] for u in element_xyz]; symbol = [u[0] for u in element_xyz]
        coords.append(coord); symbols.append(symbol)
        os.chdir(cwd)

np.savez('alldata.npz',coords=coords,symbols=symbols,confs=confs)

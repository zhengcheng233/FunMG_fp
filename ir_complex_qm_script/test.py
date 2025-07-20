#!/usr/bin/env python
from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np 

coord = []; symbol = []
with open('frame_1.com','r') as fp:
    lines = fp.readlines()[3:]
    for line in lines:
        line = line.strip().split()
        if len(line) == 4:
            symbol.append(line[0])
            coord.append([float(line[1]),float(line[2]),float(line[3])])

species = ['C']; r_cut = 10; n_max = 5; l_max = 4

soap = SOAP(species=species,
            periodic=False,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max)

mol = Atoms(symbol,positions=coord)
soap = soap.create(mol)

def dis_mat(desc):
    dis_mat = np.zeros((len(desc),len(desc)))
    for ii in range(len(desc)):
        for jj in range(len(desc)):
            tmp = np.linalg.norm((desc[ii] - desc[jj]))
            #tmp = np.dot(desc[ii],desc[jj])
            dis_mat[ii,jj] = tmp
    return dis_mat

dis = dis_mat(soap)
for ii in range(len(dis)):
    print(np.where(dis[ii]<0.01)[0])

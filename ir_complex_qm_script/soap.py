#!/usr/bin/env python
"""
1. generate soap descriptors to describe Ir atoms environmnets for each complexs
2. calculate different atom pair distances
3. using fps to sorted the sample order
"""
from dscribe.descriptors import SOAP

def calc_fp(coord,symbol,param):
    #with open('cur.json','r') as fp:
    #    param = json.load(fp)
    species = param['species']; r_cut = param['r_cut']
    n_max = param['n_max']; l_max = param['l_max']
    
    soap = SOAP(
         species=species,
         periodic=False,
         r_cut=r_cut,
         n_max=n_max,
         l_max=l_max,
    )
    #coord = []; symbol = []
    #with open(f_name,'r') as fp:
    #    for line in fp:
    #        line = line.strip().split()
    #        if len(line) == 4:
    #            coord.append([float(line[1]),float(line[2]),float(line[3])])
    #            symbol.append(line[0])
    mol = Atoms(symbol,positions=coord)
    soap_Ir = soap.create(mol,positions=[0])[0]
    return soap_Ir

def calc_dis(idx_0,descriptors):
    des0 = descriptors[idx_0]
    dis = np.sqrt(np.sum((descriptors - desc0)**2,axis=1))
    return dis



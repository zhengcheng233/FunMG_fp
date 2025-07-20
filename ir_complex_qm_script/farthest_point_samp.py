#!/usr/bin/env python
"""
using farthest point sampling method to obtain the oled qm calculation order
"""
import numpy as np
from soap import calc_fp, calc_dis
import sys
import json
from glob import glob 
import ase

atomic_n = ase.atom.atomic_numbers
n_step = int(sys.argv[1])

# step 1: divite the lib according to the elements type in complex 
def data_divide():
    def element_types_extract(symbol):
        ele = sorted(list(set(symbol)))
        ele_type = ''
        for uu in ele:
            ele_type += uu
        return ele_type
    
    def proton_num(symbol):
        proton_n = 0 
        for sym in symbol[1:]:
            proton_n += atomic_n[sym]
        return proton_n
  
    data = np.load('oled_lib.npz',allow_pickle=True)
    coords = data['coords']; symbols = data['symbols']
    homos = data['homo']; lumos = data['lumo']
    reasons = data['reasons']; ligand_idxs = data['ligand_idxs']
    
    elements = []; elements_set = []
    for symbol in symbols:
        ele = element_types_extract(symbol)
        elements.append(ele)
        if ele not in elements_set:
            elements_set.append(ele)
    
    data_sub = {}
    for ele in elements_set:
        data_sub[ele] = {'coords':[],'symbols':[],'homo':[],'lumo':[],'ligand_idx':[],'proton_n':[]} 

    for idx in range(len(coords)):
        coord = coords[idx]; symbol = symbols[idx]; homo = homos[idx]; lumo = lumos[idx] 
        ligand_idx = ligand_idxs[idx]
        ele = element_types_extract(symbol)
        proton_n = proton_num(symbol)
        data_sub[ele]['coords'].append(coord); data_sub[ele]['symbols'].append(symbol)
        data_sub[ele]['homo'].append(homo); data_sub[ele]['lumo'].append(lumo)
        data_sub[ele]['ligand_idx'].append(ligand_idx); data_sub[ele]['proton_n'].append(proton_n)

    for ele in elements_set:
        symbol = [u for u in ele]
        cur_job = {'species': symbol, 'r_cut': 15., 'n_max': 6, 'l_max': 4}
        os.system('mkdir -p '+str(ele))
        np.savez('./'+str(ele)+'/oled_lib.npz',data_sub[ele])
        with open('./'+str(ele)+'/cur.json','w') as fp:
            json.dump(cur_job, fp, indent = 4)

if n_step == 1:
    data_divide()

# step 2: calculate each Ir atom's soap and calculate the atom's pair distances
def gen_soap(f_name):
    data = np.load(f_name+'/oled_lib.npz')
    coords = data['coords']; symbols = data['symbols']
    ligand_idx = data['ligand_idx']
    with open(f_name+'/cur.json','r') as fp:
        param = json.load(fp)
    descriptors = []
    for cc,ss in zip(coords,symbols):
        soap = calc_fp(cc,ss,param)
        descriptors.append(soap)
    np.savez(f_name+'/desc.npz',ligand_idx = ligand_idx, soap = descriptors)

def gen_dis(f_name):
    descriptors = np.load(f_name+'/desc.npz',allow_pickle=True)['soap']
    pair_dis = []
    for idx in range(len(descriptors)):
        dis = calc_dis(idx, descriptors)
        pair_dis.append(dis)
    np.savez(f_name+'/'+str(idx)+'_dist.npz',pair_dis)

if n_step == 2:
    f_dirs = glob('./*/oled_lib.npz')
    for f_name in f_dirs:
        f_name = f_name.split('/')[-2]
        gen_soap(f_name)
        gen_dis(f_name) 

# step 3: farthest point sampling
def min_dis(dis_mat, set_A, idx):
    dis = dis_mat[idx,set_A]
    return np.min(dis)

def fps(dis_mat,set_A,set_B):
    """set_A, the chosen samples set; set_B, the remain samples set; dis_arr, the minial distance on each point    
    but we need to consider the num of atoms"""
    # if set_A is null, we need choose one element from B to A 
    if len(set_A) == 0:
        set_A.append(set_B[0])
    else:
        del set_B[0]
    
    dis_arr = []
    for idx in set_B:
        dis_arr.append(min_dis(dis_mat, set_A, idx))
    iter_num = len(set_B)
    for ii in range(iter_num):
        ord_min = np.argmin(dis_arr); idx_min = set_B[ord_min]
        set_A.append(idx_min)
        del dis_arr[ord_min]
        del set_B[ord_min]
        dis_tmp = min_dis(dis_mat, set_B, idx_min) 
        dis_arr = [min(a,b) for a, b in zip(dis_arr, dis_tmp)]
    return set_A, set_B

def atom_sort_fps(dis_mat, ligand_idxs, proton_nums, n_batch):
    """first divide the dataset according to the number of heavy atoms then using fps method to sorted the fps order
    final return a file, with the conf.x_y_z order"""
    
    # sort the proton num and divide into n_batch
    
    proton_num_idx = np.argsort(proton_num); res_num = len(proton_nums)%n_batch 
    proton_batch = []
    for ii in range(n_batch):
        proton_batch.append(proton_num_idx[ii*n_batch,ii*n_batch+n_batch])
    proton_batch.append(proton_num_idx[-res_num:])
    
    # with the combination of proton_batch and fps, obtain the config order
    set_A = []
    for idx,batch in enumerate(proton_batch):
        set_B = batch
        dis_mat_batch = dis[batch][:,batch]
        set_A = fps(dis_mat_batch, set_A, set_B) 
    ligand_idx_final = [ligand_idxs[uu] for uu in set_A]
    
    return ligand_idx_final

if __name__ == '__main__':
    a = 1
     

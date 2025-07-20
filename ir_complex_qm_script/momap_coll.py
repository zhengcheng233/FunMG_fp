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

iter_num = sys.argv[1]; fmt = '%06d'
#f_dirs = glob('./02.fp/conf.91_91_110')
f_dirs = glob('./conf.*')

def near_atom_search(bond,ord_0,ord_1):
    # search the atom nearby 
    bond = np.array(bond); bond_0 = bond[:,0]; bond_1 = bond[:,1]
    set_0 = []; set_1 = []; tri_idx = None; ord_0 = [ord_0]; ord_1 = [ord_1]
    for ii in range(3):
        for ord_ii in ord_0:
            ord_0 = bond_1[list(np.where(bond_0==ord_ii)[0])]
            set_0.extend(list(ord_0))
        for ord_ii in ord_1:
            ord_1 = bond_1[list(np.where(bond_0==ord_ii)[0])]
            set_1.extend(list(ord_1))
        same_ele = list(set(set_0) & set(set_1))
        if len(same_ele) > 0:
            tri_idx = same_ele[0]
            return tri_idx

def ligand_check1(coord, bond, dis_mat):
    # check whether the atoms in ligand combined with Ir atom are in the same plane
    save_label = True; coord = np.array(coord)
    c0 = coord[0]; ligand = _get_atom(bond); dis = dis_mat[0][ligand]
    ord_0 = ligand[np.argsort(dis)[0]]; ord_1 = ligand[np.argsort(dis)[1]]
    ord_2 = near_atom_search(bond, ord_0, ord_1)
    c1 = coord[ord_0]; c2 = coord[ord_1]; c3 = coord[ord_2]
    vec0 = c0 - c1; vec1 = c1 - c2; vec2 = c2 - c1; vec3 = c3 - c2
    cross_vec_0 = np.cross(vec0, vec1); cross_vec_1 = np.cross(vec2, vec3)
    cross_vec_0 = cross_vec_0 / np.linalg.norm(cross_vec_0)
    cross_vec_1 = cross_vec_1 / np.linalg.norm(cross_vec_1)
    angle = np.arccos(np.abs(np.dot(cross_vec_0, cross_vec_1)))
    ###### have problem
    if angle > 0.088 and angle < 3.1415 - 0.088:
        save_label = False
    return save_label, c1, c2

def angle_search(c0,c1,c2,c3,c4,c5):
    c0 = c0/np.linalg.norm(c0); c1 = c1/np.linalg.norm(c1)
    c2 = c2/np.linalg.norm(c2); c3 = c3/np.linalg.norm(c3)
    c4 = c4/np.linalg.norm(c4); c5 = c5/np.linalg.norm(c5)
    theta = 0
    theta0 = np.arccos(np.dot(c0,c1))
    if theta < theta0:
        theta = theta0
    theta1 = np.arccos(np.dot(c0,c2))
    if theta < theta1:
        theta = theta1
    theta2 = np.arccos(np.dot(c0,c3))
    if theta < theta2:
        theta = theta2
    theta3 = np.arccos(np.dot(c0,c4))
    if theta < theta3:
        theta = theta3
    theta4 = np.arccos(np.dot(c0,c5))
    if theta < theta4:
        theta = theta4
    assert(theta < 3.15)
    assert(theta > 1.72)
    return theta
    

def ligand_check2(c0,c1,c2):
    reason = True
    c0_0 = c0[0]; c0_1 = c0[1]; c1_0 = c1[0]; c1_1 = c1[1]; c2_0 = c2[0]; c2_1 = c2[1]
    theta0 = angle_search(c0_0,c0_1,c1_0,c1_1,c2_0,c2_1)
    if theta0 < 2.96:
        reason = False
    theta0 = angle_search(c0_1,c0_0,c1_0,c1_1,c2_0,c2_1)
    if theta0 < 2.96:
        reason = False
    theta0 = angle_search(c1_0,c0_1,c0_0,c1_1,c2_0,c2_1)
    if theta0 < 2.96:
        reason = False
    theta0 = angle_search(c1_1,c0_1,c1_0,c0_0,c2_0,c2_1)
    if theta0 < 2.96:
        reason = False
    return reason 

def tight_stand(f_name,coord0,coord1,symbol):
    save_label = True
    bond_0, bond_length_0 = topo_bond(os.path.join(f_name,'topo_0.txt'))
    bond_1, bond_length_1 = topo_bond(os.path.join(f_name,'topo_1.txt'))
    bond_2, bond_length_2 = topo_bond(os.path.join(f_name,'topo_2.txt'))
    bond_idx_0, bond_idx_1, bond_idx_2 = bond_read(os.path.join(f_name,'bond.txt'))
    dis_mat0 = mol_distance(coord0); dis_mat1 = mol_distance(coord1)
    reason = ligand_check(dis_mat0, bond_0, bond_length_0)
    if reason == False:
        save_label = False
    reason = ligand_check(dis_mat0, bond_1, bond_length_1)
    if reason == False:
        save_label = False
    reason = ligand_check(dis_mat0, bond_2, bond_length_2)
    if reason == False:
        save_label = False
    reason1 = bond_check2(dis_mat0, bond_0, bond_1, bond_2, coord0, symbol)
    if reason1 == False:
        save_label = False
    save_label0 = save_label

    reason, c0_0, c0_1 = ligand_check1(coord1, bond_0, dis_mat1)
    if reason == False:
        save_label = False
    reason, c1_0, c1_1 = ligand_check1(coord1, bond_1, dis_mat1)
    if reason == False:
        save_label = False
    reason, c2_0, c2_1 = ligand_check1(coord1, bond_2, dis_mat1)
    if reason == False:
        save_label = False
    c_Ir = coord1[0]
    reason = ligand_check2([c0_0-c_Ir,c0_1-c_Ir],[c1_0-c_Ir,c1_1-c_Ir],[c2_0-c_Ir,c2_1-c_Ir])
    if reason == False:
        save_label = False
    return save_label, save_label0

def dataread(f_plqy, f_edme, f_homo, f_inp):
    """
    read data from momap file
    """

    coord = []; symbol = []
    plqy = None; edme = None; homo = None; lumo = None; e_ad = None; read_lumo = False
    plqy_data = json.load(open(f_plqy))['0.0']
    plqy = plqy_data[0]/(plqy_data[0] + max(plqy_data[1],0))
    edme_data = json.load(open(f_edme))['1.0']
    edme = edme_data[2]; e_ad = edme_data[0]
    fr = open(f_homo); lines = fr.readlines(); fr.close()
    for ii in range(len(lines)-1):
        line0 = lines[ii].strip().split()
        line1 = lines[ii+1].strip().split()
        if len(line0) > 3:
            if line0[0] == 'Alpha' and line0[1] == 'occ.' and line0[2] == 'eigenvalues':
                if len(line1) == 5:
                    assert(line1[1] == 'occ.')
                    homo = float(line0[-1]); lumo = float(line1[-1])
                else:
                    if len(line0) > 5:
                        homo = float(line0[-2]); lumo = float(line0[-1])
    assert(homo != None)
    element_xyz = read_init_xyz(f_homo)
    coord_opt = [u[1:] for u in element_xyz]
    with open(f_inp) as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 4:
                symbol.append(line[0]); coord.append([float(line[1]),float(line[2]),float(line[3])])

    return plqy, e_ad, edme, homo, lumo, coord, symbol, coord_opt

momap_data = {'coord':[],'symbol':[],'conf_idx':[],'plqy':[],'e_ad':[],'edme':[],'homo':[],'lumo':[]}
momap_data_loose = {'coord':[],'symbol':[],'conf_idx':[],'plqy':[],'e_ad':[],'edme':[],'homo':[],'lumo':[]}
for f_name in f_dirs:
    f_0 = os.path.basename(f_name)
    f_plqy = os.path.join(f_name,'plqy.json') 
    f_edme = os.path.join(f_name,'data.json')
    f_homo = os.path.join(f_name,'t1-opt.log')
    f_inp = os.path.join(f_name,'input.com')
    if os.path.isfile(f_plqy) is not True:
        continue
    plqy, e_ad, edme, homo, lumo, coord, symbol, coord_opt = dataread(f_plqy, f_edme, f_homo, f_inp)
    tight_cri, tight_cri1 = tight_stand(f_name, coord, coord_opt, symbol)
    if math.isnan(plqy):
        print('nan value')
        tight_cri1 = False
    if tight_cri1 == True:
        if tight_cri == True:
            momap_data['coord'].append(coord); momap_data['symbol'].append(symbol)
            momap_data['conf_idx'].append(f_0); momap_data['plqy'].append(plqy)
            momap_data['e_ad'].append(e_ad); momap_data['edme'].append(edme)
            momap_data['homo'].append(homo); momap_data['lumo'].append(lumo)
        momap_data_loose['coord'].append(coord); momap_data_loose['symbol'].append(symbol)
        momap_data_loose['conf_idx'].append(f_0); momap_data_loose['plqy'].append(plqy)
        momap_data_loose['e_ad'].append(e_ad); momap_data_loose['edme'].append(edme)
        momap_data_loose['homo'].append(homo); momap_data_loose['lumo'].append(lumo) 
    else:
        print(f_name)

f_name = fmt %(int(iter_num))
np.savez('iter.'+f_name+'.npz',coord=momap_data['coord'],symbol=momap_data['symbol'],conf_idx=momap_data['conf_idx'],\
         plqy=momap_data['plqy'],e_ad=momap_data['e_ad'],edme=momap_data['edme'],homo=momap_data['homo'],lumo=momap_data['lumo'])

np.savez('iter.'+f_name+'_loose.npz',coord=momap_data_loose['coord'],symbol=momap_data_loose['symbol'],conf_idx=momap_data_loose['conf_idx'],\
         plqy=momap_data_loose['plqy'],e_ad=momap_data_loose['e_ad'],edme=momap_data_loose['edme'],homo=momap_data_loose['homo'],lumo=momap_data_loose['lumo'])


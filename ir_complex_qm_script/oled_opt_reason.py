#!/usr/bin/env python
"""
using the bonded idxs to judge whether the structure are reasonable after structure optimization 
here, we also need to know the bonded atom idx for each molecule
"""
import sys
import numpy as np
from scipy.spatial.distance import cdist
from glob import glob
import os

def read_file():
    homo = None; lumo = None
    coord = []; bonder = []; symbol = []
    
    with open('./log','r') as fp:
        for line in fp:
            if '(HOMO)' in line:
                line = line.strip().split()
                homo = float(line[-2])
            if '(LUMO)' in line:
                line = line.strip().split()
                lumo = float(line[2])

    with open('./xtbopt.xyz','r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 4:
                symbol.append(line[0])
                coord.append([float(line[1]),float(line[2]),float(line[3])])

    coord = np.array(coord)
    return symbol, coord, homo, lumo

def topo_bond(f_name):
    bonds = []; bonds_lengths = {}
    with open(f_name,'r') as fp:
        for idx, line in enumerate(fp):
            if idx > 0:
                line = line.strip().split()
                bonds.append([int(line[0]),int(line[1])])
                bonds_lengths[tuple([int(line[0]),int(line[1])])] = (float(line[2]))
    return bonds, bonds_lengths

def mol_distance(coord):
    dis_mat = cdist(coord,coord,'euclidean')
    return dis_mat

def ligand_check(dis_mat, bonds, bonds_length):
    reason = True
    for bond in bonds:
        dis = dis_mat[bond[0],bond[1]]
        bond_length = bonds_length[tuple(bond)]
        if dis > bond_length * 1.2:
            reason = False
    return reason 

def _get_atom(bonds):
    atom_idx = []
    for bond in bonds:
        atom_idx.extend(bond)
    atom_idx = sorted(list(set(atom_idx)))
    return atom_idx

def bond_check(dis_mat, bond_idx_0, bond_idx_1, bond_idx_2, bond_0, bond_1, bond_2, coord):
    reason = True
    bond_dis_0_0 = dis_mat[0,bond_idx_0[0]]; bond_dis_0_1 = dis_mat[0,bond_idx_0[1]]
    bond_dis_1_0 = dis_mat[0,bond_idx_1[0]]; bond_dis_1_1 = dis_mat[0,bond_idx_1[1]]
    bond_dis_2_0 = dis_mat[0,bond_idx_2[0]]; bond_dis_2_1 = dis_mat[0,bond_idx_2[1]]
    ligand_0 = _get_atom(bond_0); ligand_1 = _get_atom(bond_1); ligand_2 = _get_atom(bond_2)
    dis_0 = np.sqrt(np.sum((coord[0] - coord[ligand_0])**2,axis=1))
    dis_1 = np.sqrt(np.sum((coord[0] - coord[ligand_1])**2,axis=1))
    dis_2 = np.sqrt(np.sum((coord[0] - coord[ligand_2])**2,axis=1))
    dis_0 = dis_0[np.argsort(dis_0)][2]; dis_1 = dis_1[np.argsort(dis_1)][2]; dis_2 = dis_2[np.argsort(dis_2)][2]
    if bond_dis_0_0 > dis_0 or bond_dis_0_1 > dis_0:
        reason = False
    if bond_dis_1_0 > dis_1 or bond_dis_1_1 > dis_1:
        reason = False
    if bond_dis_2_0 > dis_2 or bond_dis_2_1 > dis_2:
        reason = False
    return reason 

def dot_calc(ord_0, ligand_0, coord, symbol):
    reason = True
    ligand0_1 = ligand_0[ord_0[0]]; ligand0_2 = ligand_0[ord_0[1]]
    if symbol[ligand0_1] == 'H' or symbol[ligand0_2] == 'H':
        reason = False
    vec_0 = coord[0] - coord[ligand0_1]; vec_1 = coord[0] - coord[ligand0_2]
    cross_vec = np.cross(vec_0, vec_1)
    return reason, cross_vec

def dot_calc2(ord_0, ligand_0, coord, symbol):
    coord = np.array(coord)
    reason = True
    ligand0_1 = ligand_0[ord_0[0]]; ligand0_2 = ligand_0[ord_0[1]]
    if symbol[ligand0_1] == 'H' or symbol[ligand0_2] == 'H':
        reason = False
    dis_1 = np.sum((coord - coord[ligand0_1])**2,axis=1)
    dis_2 = np.sum((coord - coord[ligand0_2])**2,axis=1)
    set_1 = list(np.argsort(dis_1)[1:3]); set_2 = list(np.argsort(dis_2)[1:3])
    
    if ligand0_2 in set_1 or ligand0_1 in set_2:
        reason = False
    vec_0 = coord[0] - coord[ligand0_1]; vec_1 = coord[0] - coord[ligand0_2]
    cross_vec = np.cross(vec_0, vec_1)
    return reason, cross_vec

def bond_check2(dis_mat, bond_0, bond_1, bond_2, coord, symbol):
    reason = True
    ligand_0 = _get_atom(bond_0); ligand_1 = _get_atom(bond_1); ligand_2 = _get_atom(bond_2)
    dis_0 = dis_mat[0][ligand_0]; dis_1 = dis_mat[0][ligand_1]; dis_2 = dis_mat[0][ligand_2]
    ord_0 = np.argsort(dis_0); ord_1 = np.argsort(dis_1); ord_2 = np.argsort(dis_2)
    reason_lig, cross_vec_0 = dot_calc2(ord_0, ligand_0, coord, symbol)  
    if reason_lig == False:
        reason = False
    reason_lig, cross_vec_1 = dot_calc2(ord_1, ligand_1, coord, symbol)
    if reason_lig == False:
        reason = False
    reason_lig, cross_vec_2 = dot_calc2(ord_2, ligand_2, coord, symbol)
    if reason_lig == False:
        reason = False
    length_0 = np.sqrt(np.sum(cross_vec_0**2))
    length_1 = np.sqrt(np.sum(cross_vec_1**2))
    length_2 = np.sqrt(np.sum(cross_vec_2**2))
    angle_0 = np.arccos(np.dot(cross_vec_0,cross_vec_1)/length_0/length_1)
    angle_1 = np.arccos(np.dot(cross_vec_0,cross_vec_2)/length_0/length_2)
    angle_2 = np.arccos(np.dot(cross_vec_1,cross_vec_2)/length_1/length_2)
    if angle_0 > 1.5707 * 1.11 or angle_0 < 1.5707 * 0.89:
        reason = False
    if angle_1 > 1.5707 * 1.11 or angle_1 < 1.5707 * 0.89:
        reason = False
    if angle_2 > 1.5707 * 1.11 or angle_2 < 1.5707 * 0.89:
        reason = False
    return reason   

def bond_check1(dis_mat, bond_0, bond_1, bond_2, coord, symbol):
    # adjust whether stereo configuration is reasonable
    reason = True
    ligand_0 = _get_atom(bond_0); ligand_1 = _get_atom(bond_1); ligand_2 = _get_atom(bond_2)
    dis_0 = dis_mat[0][ligand_0]; dis_1 = dis_mat[0][ligand_1]; dis_2 = dis_mat[0][ligand_2]
    ord_0 = np.argsort(dis_0); ord_1 = np.argsort(dis_1); ord_2 = np.argsort(dis_2)
    reason_lig, cross_vec_0 = dot_calc(ord_0, ligand_0, coord, symbol)
    if reason_lig == False:
        reason = False
    reason_lig, cross_vec_1 = dot_calc(ord_1, ligand_1, coord, symbol)
    if reason_lig == False:
        reason = False
    reason_lig, cross_vec_2 = dot_calc(ord_2, ligand_2, coord, symbol)
    if reason_lig == False:
        reason = False
    length_0 = np.sqrt(np.sum(cross_vec_0**2))
    length_1 = np.sqrt(np.sum(cross_vec_1**2))
    length_2 = np.sqrt(np.sum(cross_vec_2**2))
    angle_0 = np.arccos(np.dot(cross_vec_0,cross_vec_1)/length_0/length_1) 
    angle_1 = np.arccos(np.dot(cross_vec_0,cross_vec_2)/length_0/length_2)
    angle_2 = np.arccos(np.dot(cross_vec_1,cross_vec_2)/length_1/length_2)
    if angle_0 > 1.5707 * 1.11 or angle_0 < 1.5707 * 0.88:
        reason = False
    if angle_1 > 1.5707 * 1.11 or angle_1 < 1.5707 * 0.88:
        reason = False
    if angle_2 > 1.5707 * 1.11 or angle_2 < 1.5707 * 0.88:
        reason = False
    return reason

def reasonable_judge(dis_mat, symbol, bond_0, bond_length_0, bond_1, \
    bond_lenth_1, bond_2, bond_length_2, bond_idx_0, bond_idx_1, bond_idx_2, coord, dir0):
    # verity whether the optimized oled structure is reasonable
    reasonable = True
    # first check the ligand is reasonable
    reason = ligand_check(dis_mat, bond_0, bond_length_0)
    if reason == False:
        reasonable = False
    reason = ligand_check(dis_mat, bond_1, bond_length_1)
    if reason == False:
        reasonable = False
    reason = ligand_check(dis_mat, bond_2, bond_length_2)
    if reason == False:
        reasonable = False
    # second check the bonded atom is reasonable
    reason1 = bond_check(dis_mat, bond_idx_0, bond_idx_1, bond_idx_2, bond_0, bond_1, bond_2, coord)
    if reason1 == False:
        #print('******sth is wrong************')
        #print(dir0)
        reason1 = bond_check1(dis_mat, bond_0, bond_1, bond_2, coord, symbol)
    if reason1 == False:
        reasonable = False
    return reasonable

def bond_read(f_name):
    bonds = []
    with open(f_name,'r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 2:
                bonds.append([int(line[0]),int(line[1])])
    return bonds[0], bonds[1], bonds[2]


def gaussianread(f_name):
    symbol = []; coord = []
    with open(f_name,'r') as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) == 4:
                coord.append([float(line[1]),float(line[2]),float(line[3])])
                symbol.append(line[0])
    coord = np.array(coord)
    return coord, symbol

#======================================================================================================
if __name__ == '__main__':
    num = sys.argv[1]
    opt_dirs = glob('./viturl_simp/conf.'+str(num)+'_*')
    cwd_ = os.getcwd()
    coords = []; symbols = []; homos = []; lumos = []; reasons = []; ligand_idxs = []
    for dir0 in opt_dirs:
        if os.path.isdir(dir0) is not True:
            os.chdir(cwd_)
            continue
        os.chdir(dir0)
        if os.path.isfile('topo_0.txt') is not True or os.path.isfile('topo_1.txt') is not True or \
            os.path.isfile('topo_2.txt') is not True or os.path.isfile('bond.txt') is not True or \
            os.path.isfile('complex_opt.com') is not True:
            os.chdir(cwd_)
            continue
        f_name = os.path.basename(dir0).split('.')[-1].split('_')
        ligand_idxs.append([f_name[0],f_name[1],f_name[2]])
        bond_0, bond_length_0 = topo_bond('topo_0.txt')
        bond_1, bond_length_1 = topo_bond('topo_1.txt')
        bond_2, bond_length_2 = topo_bond('topo_2.txt')
        bond_idx_0, bond_idx_1, bond_idx_2 = bond_read('bond.txt')
        coord, symbol = gaussianread('complex_opt.com')
        dis_mat = mol_distance(coord) 
        reason = reasonable_judge(dis_mat, symbol, bond_0, bond_length_0, bond_1, bond_length_1,\
                bond_2, bond_length_2, bond_idx_0, bond_idx_1, bond_idx_2, coord, dir0)
        with open('reason.txt','w') as fp:
            if reason == True:
                fp.write('1'+'\n')
                reasons.append(1)
            else:
                fp.write('0'+'\n')
                reasons.append(0)
        symbol,coord,homo,lumo = read_file()
        coords.append(coord); symbols.append(symbol); homos.append(homo); lumos.append(lumo)
        
        os.chdir(cwd_)
    os.chdir(cwd_)
    os.chdir('pretraindata')
    np.savez('oled_'+str(num)+'_data.npz',coords=coords,symbols=symbols,homo=homos,lumo=lumos,reasons=reasons,ligand_idxs=ligand_idxs)
    os.chdir(cwd_)
    

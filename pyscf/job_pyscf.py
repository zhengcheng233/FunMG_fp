#!/usr/bin/env python 
from typing import List, Dict, Any
import subprocess
import logging
from ase.data import chemical_symbols, atomic_numbers
import numpy as np 
import json 
from pyscf import gto
from pyscf.geomopt.geometric_solver import optimize
from pyscf import tdscf
import pyscf 
'''
包含main函数，解析参数；make_fp函数：批处理初始文件，生成s0的input.com
run_fp函数：根据任务不同运行相应的任务：包括s0, s1, t1 的结构优化，单点计算，光谱生成
post_fp函数：根据任务不同，处理输出文件，生成最终结果，可能是json文件，可能是光谱数据
'''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gaussian(x, mu, fwhm, amplitude):
    sigma = fwhm /(2 * np.sqrt(2 * np.log(2)))  # 展宽系数
    return 28700 * amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def convert_to_angstrom(atom_list:List)->Any:
    new_atom_list = []
    for atom in atom_list:
        symbol = atom[0]
        coord_angstrom = np.array(atom[1]) * 0.52917724
        new_atom_list.append((symbol, list(coord_angstrom)))
    return new_atom_list

def make_fp(jparm:Dict, mparm:Dict, data:Dict)->Any:
    # 生成mol对象
    input_name = jparm.get('input_name','input.xyz')
    method = jparm.get('method','b3lyp EmpiricalDispersion=GD3BJ')
    basis = jparm.get('basis','def2svp')
    charge = jparm.get('charge','0')
    multiplicity = jparm.get('multiplicity','1') - 1
    nproc = mparm.get('nproc','32')
    mem = mparm.get('mem','64GB')

    # pyscf目前似乎不支持tddft + d3校正的计算，因此method需要忽略d3相关参数
    if len(method.split()) > 1:
        method = method.split()[0]

    # 读取input文件, 目前只支持xyz文件
    coord = []; symbol = []
    with open(input_name, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            l = line.strip().split()
            if len(l) == 4:
                symbol.append(l[0])
                coord.append([float(l[1]), float(l[2]), float(l[3])])
    
    atom_list = []
    for i in range(len(symbol)):
        atom_list.append((symbol[i], coord[i]))
    
    # 构造mol对象
    mol = gto.Mole()
    mol.atom = atom_list
    mol.basis = basis
    mol.charge = charge 
    mol.spin = multiplicity
    mol.build()
    # 由于任何任务的基础都是s0结构优化，所以s0的opt任务直接在make_fp处完成
    # step 1 运行pyscf实现s0的结构优化
    # 构造mol的波函数、哈密顿等物理量
    mf = mol.RKS()
    mf.xc = method 
    # 构造结构优化器（pyscf只有该优化器能返回opt是否converged....）
    g_scan = mf.Gradients().as_scanner()
    # 运行s0的结构优化
    opt = g_scan.optimizer(solver='geomeTRIC').run()
    conv = opt.converged # 结构优化是否收敛判据
    # 从优化器中获取优化后的mol对象
    optimized_mol = opt.mol 
    # 获取优化后的元素&坐标信息 
    atom_list = convert_to_angstrom(optimized_mol.atom)# 坐标&元素 in bohr 

    # 基于优化后的mol对象，计算单点，获取电子性质
    mf_optimized = optimized_mol.RKS()
    mf_optimized.xc = method 
    mf_optimized.kernel()
    # 获取基态能量，homo, lumo能量，计算h-l gap 
    ground_e = mf_optimized.e_tot 
    mo_energy = mf_optimized.mo_energy
    homo_energy = mo_energy[mf_optimized.mo_occ > 0][-1]
    lumo_energy = mo_energy[mf_optimized.mo_occ == 0][0]
    data['converge'] = conv; data['s0_ground_e'] = ground_e
    data['homo'] = homo_energy * 27.2114
    data['lumo'] = lumo_energy * 27.2114
    data['HL_gap'] = (lumo_energy - homo_energy) * 27.2114
    data['atom_list'] = atom_list 
    return data 

def run_fp(jparm:Dict, mparm:Dict, data:Dict)->Any:
    input_name = jparm.get('input_name','input.xyz')
    method = jparm.get('method','b3lyp')
    basis = jparm.get('basis','def2svp')
    charge = jparm.get('charge','0')
    multiplicity = jparm.get('multiplicity','1') - 1 
    nproc = mparm.get('nproc','32')
    mem = mparm.get('mem','64GB')
    task_name = jparm.get('task','s0')
    
    # 由于pyscf目前并不支持tddft的dispersion计算，所以暂时不支持dispersion计算
    if len(method.split()) > 1:
        method = method.split()[0]
    # 构造mol&mf对象
    mol = gto.Mole()
    mol.atom = data['atom_list']
    mol.basis = basis
    mol.charge = charge 
    mol.spin = multiplicity
    mol.build()
    mf = mol.RKS()
    mf.xc = method
    mf.kernel()


    # 在获得s0优化后的结构后，run_fp存在多个任务包括：
    # 谱学任务：吸收光谱 & 吸收能计算；
    # 光物理任务s1：发射能 & 重整能计算 (顺便计算吸收光谱&吸收能) & s0s1能量差
    # 氧化性任务：垂直电离能、垂直亲和能、绝热电离能、绝热亲和能、电子重组能、空穴重整能
    if task_name == 'absorption_elec_spec':
        # 吸收光谱计算，简单的s1单点计算
        mf_td = tdscf.TDDFT(mf)
        mf_td.nstates = 5 # 30 一般对于benzene以上取30合适
        td_result = mf_td.kernel()
        # 分析波函数获取性质
        mf_td.analyze()
        fs = mf_td.oscillator_strength(gauge='length') # 振子强度
        e_abs = td_result[0] * 27.2114 # 所有态积分能
        # 获取吸收能
        abs_energy_max = e_abs[np.argmax(fs)]
        # 获取光谱
        fs = np.array(fs); e_abs = np.array(e_abs)
        min_energy = 1.24; max_energy = 5.64
        energy_range = np.linspace(min_energy, max_energy, 1000)
        fwhm = 0.5
        spectrum = np.zeros_like(energy_range)
        for energy, strength in zip(e_abs, fs):
            spectrum += gaussian(energy_range, energy, fwhm, strength)
        data['abs_energy_max'] = abs_energy_max
        data['spectrum_y'] = list(spectrum)
        data['spectrum_x'] = list(energy_range)

    elif task_name == 's1':
        # 吸收光谱计算，简单的s1单点计算
        mf_td = tdscf.TDDFT(mf)
        mf_td.nstates = 5 # 30 一般对于benzene以上取30合适
        td_result = mf_td.kernel()
        # 分析波函数获取性质
        mf_td.analyze()
        fs = mf_td.oscillator_strength(gauge='length') # 振子强度
        e_abs = td_result[0] * 27.2114 # 所有态积分能
        # 获取吸收能
        abs_energy_max = e_abs[np.argmax(fs)]
        # 获取光谱
        fs = np.array(fs); e_abs = np.array(e_abs)
        min_energy = 1.24; max_energy = 5.64
        energy_range = np.linspace(min_energy, max_energy, 1000)
        fwhm = 0.5
        spectrum = np.zeros_like(energy_range)
        for energy, strength in zip(e_abs, fs):
            spectrum += gaussian(energy_range, energy, fwhm, strength)
        data['abs_energy_max'] = abs_energy_max
        data['spectrum_y'] = list(spectrum)
        data['spectrum_x'] = list(energy_range)
        
        data['s0_td_e'] = td_result[0][0] + data['s0_ground_e']
        # 基于s0的结构完成tddft下的结构优化
        mf_td.nstates = 5 
        g_scan_td = mf_td.Gradients().as_scanner()
        mf_td_opt = g_scan_td.optimizer(solver='geomeTRIC').run()
        conv = mf_td_opt.converged
        if data['converge'] == True:
            data['converge'] = conv 
        td_opt_mol = mf_td_opt.mol
        atom_info = convert_to_angstrom(td_opt_mol.atom)
        data['s1_atom_list'] = atom_info 
        # 计算优化结构下的tddft
        mf_td_opt_mol = td_opt_mol.RKS()
        mf_td_opt_mol.xc = method
        mf_td_opt_mol.kernel()
        mf_td_opt_mol = tdscf.TDDFT(mf_td_opt_mol)
        mf_td_opt_mol.nstates = 5
        s1_result = mf_td_opt_mol.kernel()
        data['s1_td_e'] = mf_td_opt_mol.e_tot[0] # s1优化结构的第一激发能 in hartree
        data['s1_ground_e'] = data['s1_td_e'] - s1_result[0][0] # s1优化结构的基态能量 in hartree
        data['emi_energy'] = (data['s1_td_e'] - data['s1_ground_e']) * 27.2114 # 发射能 in eV 
        data['emi_reorg_energy'] =  (data['s0_td_e'] - data['s1_td_e']) + \
                                    (data['s1_ground_e'] - data['s0_ground_e']) 
        data['emi_reorg_energy'] = data['emi_reorg_energy'] * 27.2114 #发射重组能 in eV
        data['s0s1_e'] = (data['s1_td_e'] - data['s0_ground_e']) * 27.2114 # in eV
        return data 

    elif task_name == 'ionization':
        # 氧化还原性质计算，计算电离能相关，与发射类似
        # 计算基态优化结构下阳离子能量
        mol.charge = charge + 1
        mol.spin = 1 # 阴阳离子2s必然=1 
        mf = mol.RKS()
        mf.xc = method 
        result = mf.kernel()
        data['s0_ion_e'] = result
        # 优化阳离子
        g_scan_td = mf.Gradients().as_scanner()
        mf_opt = g_scan_td.optimizer(solver='geomeTRIC').run()
        conv = mf_opt.converged
        if data['converge'] == True:
            data['converge'] = conv
        # 获取阳离子优化后坐标
        ion_opt_mol = mf_opt.mol
        atom_info = convert_to_angstrom(ion_opt_mol.atom)
        data['ion_atom_list'] = atom_info 
        # 计算阳离子能量
        mf_ion_opt_mol = ion_opt_mol.RKS()
        mf_ion_opt_mol.xc = method 
        ion_opt_result = mf_ion_opt_mol.kernel()
        data['ion_ion_e'] = ion_opt_result
        # 计算阳离子优化结构下的中性能量
        ion_opt_mol.charge = charge
        ion_opt_mol.spin = 0 
        mf = ion_opt_mol.RKS()
        mf.xc = method 
        result = mf.kernel()
        data['ion_ground_e'] = result
        data['vertical_ion_e'] = (data['s0_ion_e'] - data['s0_ground_e']) * 27.2114 
        data['adiabatic_ion_e'] = (data['ion_ion_e'] - data['s0_ground_e']) * 27.2114 
        data['ion_reorg_e'] = data['s0_ion_e'] - data['ion_ion_e'] + \
                              data['ion_ground_e'] - data['s0_ground_e']
        data['ion_reorg_e'] = data['ion_reorg_e'] * 27.2114

    elif task_name == 'affinity':
        # 氧化还原性质计算，计算亲和能相关，与发射类似
        mol.charge = charge - 1
        mol.spin = 1 # 阴阳离子2s必然=1 
        mf = mol.RKS()
        mf.xc = method 
        result = mf.kernel()
        data['s0_anion_e'] = result
        # 优化阴离子
        g_scan_td = mf.Gradients().as_scanner()
        mf_opt = g_scan_td.optimizer(solver='geomeTRIC').run()
        conv = mf_opt.converged
        if data['converge'] == True:
            data['converge'] = conv
        # 获取阴离子优化后坐标
        anion_opt_mol = mf_opt.mol
        atom_info = convert_to_angstrom(anion_opt_mol.atom)
        data['anion_atom_list'] = atom_info 
        # 计算阴离子能量
        mf_anion_opt_mol = anion_opt_mol.RKS()
        mf_anion_opt_mol.xc = method 
        anion_opt_result = mf_anion_opt_mol.kernel()
        data['anion_anion_e'] = anion_opt_result
        # 计算阴离子优化结构下的中性能量
        anion_opt_mol.charge = charge
        anion_opt_mol.spin = 0 
        mf = anion_opt_mol.RKS()
        mf.xc = method 
        result = mf.kernel()
        data['anion_ground_e'] = result
        data['vertical_aff_e'] = (data['s0_ground_e'] - data['s0_anion_e']) * 27.2114 
        data['adiabatic_aff_e'] = (data['s0_ground_e'] - data['anion_anion_e']) * 27.2114 
        data['anion_reorg_e'] = data['s0_anion_e'] - data['anion_anion_e'] + \
                              data['anion_ground_e'] - data['s0_ground_e']
        data['anion_reorg_e'] = data['anion_reorg_e'] * 27.2114
    else:
        logging.error(f'Unsupported task: {task_name}')
    return data 

def post_fp(jparm:Dict, mparm:Dict)->Any:
    '''
    后处理运行任务, 由于pyscf本身提供python接口,无需后处理
    '''
    pass 
    return 

def main():
    '''
    main函数需要解析jparm, mparm参数(json文件暂定)，调用make_fp, run_fp, post_fp函数
    jparm:Dict, mparm:Dict
    eg: jparm = {'task':'s0', 'input':'input.com', 'method':'b3lyp', 'basis':'6-31g', 'charge':0, 'multiplicity':1}
    mparm = {'nproc':4, 'mem':'2GB'}
    '''
    # 解析json文件生成jparm, mparm
    # 需要传给我一个input.xyz文件
    def parse_json(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    
        # 假设JSON文件的结构是 {"jparm": {...}, "mparm": {...}}
        jparm = data.get('jparm', {})
        mparm = data.get('mparm', {})
        return jparm, mparm

    jparm, mparm = parse_json('input.json')
    # 前处理, 包括检查任务是否合法，检查输入是否存在等
    data = make_fp(jparm, mparm, {})
    # 运行任务
    data = run_fp(jparm, mparm, data)
    # 后处理, 包括生成json文件或者光��数据
    #post_fp(jparm, mparm)
    with open('data.json', 'w') as fp:
        json.dump(data, fp)
    return 

if __name__ == '__main__':
    main()
    

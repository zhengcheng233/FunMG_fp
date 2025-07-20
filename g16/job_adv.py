#!/usr/bin/env python 
from typing import List, Dict, Any
import subprocess
import fchic # 可以直接读取log信息
from ase.data import chemical_symbols, atomic_numbers
import numpy as np 
import logging 
import json
import copy
from pathlib import Path
import argparse 
import pandas as pd 
import sys 
import os 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import geometry_2_input, smi_2_geom, slurm 
from ase.data import chemical_symbols
from glob import glob
import lmdb 
import pickle 

'''
莱特任务：
1. 吸收&发射光谱模拟（TDDFT + Franck-Condon振动分析，模拟吸收和发射光谱形状和峰位）
2. 单重-三重态ISC & RISC 速率计算, soc+费米黄金法则计算ISC&RISC,预测材料是否适合TADF
3. 荧光寿命和量子效率预测，通过计算辐射跃迁率&非辐射跃迁率，估算荧光或磷光寿命及量子效率
4. 计算激发态的辐射和非辐射跃迁率，估算激发态寿命，影响发光强度和效率
拆解任务：
1. 基础物性: 发射能、重组能、吸收能、吸收跃迁偶极矩、发射跃迁偶极矩
2. 荧光激发态寿命&光致发光量子效率（PLQY）计算（不考虑系间穿越）
3. 振动分辨发射光谱计算
4. 最低单重激发态S1-最低三重激发态T1的自旋轨道耦合（SOC）计算&单重-三重态间ISC和RISC速率
5. 总和PLQY（同时考虑单重态和三重态之间的平衡以及各自弛豫速率）
参考脚本: 
'''

path_prefix=Path('/vepfs/fs_users/chensq/project/funmg/runtime_data/tasks/dft')


def gen_com(args:argparse.Namespace):
    '''csv e.g. 
       molecule_id,smiles,symbol,x,y,z
       mol1,CC(C)O,C,0.000,0.000,0.000
       mol1,CC(C)O,C,1.500,0.000,0.000
    '''
    in_pth = Path(args.in_pth, 'input.csv')
    data = pd.read_csv(in_pth)
    if 'symbol' in data.keys():
        coords, symbols, mol_names, smis = [], []
        for mol_id, group in data.groupby('molecule_id'):
            try:
                coord = group[['x', 'y', 'z']].values.tolist()
                symbol = group['symbol'].tolist()
                smi = group['smiles'].tolist()[0]
                if len(coord) == len(symbol) and len(symbol) > 0:
                    coords.append(coord)
                    symbols.append(symbol)
                    mol_names.append(mol_id)
                    smis.append(smi)
            except Exception as e:
                logging.error(f'Error processing molecule {mol_id}: {e}')
    else:
        # 如果没有坐标信息，采用rdkit转换
        coords, symbols, mol_names, smis = [], [], [], []
        for mol_id, smi in zip(data['molecule_id'], data['smiles']):
            try:
                coord, symbol, _ = smi_2_geom.smi_2_geom(smi)
                if coord is not None and symbol is not None:
                    coords.append(coord)
                    symbols.append(symbol)
                    mol_names.append(mol_id)
                    smis.append(smi)
            except Exception as e:
                logging.error(f'Error processing molecule {mol_id}: {e}')
    for coord, symbol, mol_name, smi in zip(coords, symbols, mol_names, smis):
        keywords = ['#pm6', '', 'ge', '', '0 1']
        f_name = Path(args.in_pth, f'{mol_name}', 'input.com')
        geometry_2_input.geom_2_dircom(f_name, coord, symbol, keywords)
    
def opt_calculator(args:argparse.Namespace):
    '''
    高斯软件的结构优化&频率分析，保留为单个的pkl文件
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    nproc = args.nproc; memory = args.memory; method = args.method; basis = args.basis
    charge = args.charge; multiplicity = args.multiplicity; freq = args.freq
    electronic_state = args.electronic_state
    if os.path.exists(f'{out_pth}/result.pkl'):
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}; result['Error'] = []  # 错误信息
    os.makedirs(tmp_pth, exist_ok=True)
    if electronic_state == 'S1':
        method = f'td {method}'
    if freq == True:
        keywords = [f'%chk={electronic_state.lower()}_opt.chk', f'%nproc={nproc}', f'%mem={memory}', \
               f'# {method}/{basis} opt freq', '', 'opt', '', f'{charge} {multiplicity}']
    else:
        keywords = [f'%chk={electronic_state.lower()}_opt.chk', f'%nproc={nproc}', f'%mem={memory}', \
               f'# {method}/{basis} opt', '', 'opt', '', f'{charge} {multiplicity}']
    if electronic_state == 'S0':
        coord, symbol = geometry_2_input.read_geom(Path(in_pth, 'input.com'))
        coord = np.array(coord) 
    else:
        # s1 & t1的初始结构均基于s0 opt 
        # 直接从csv中读取
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            data = pickle.load(f)
            # 读取坐标和符号
            coord = np.array(data['s0opt_coord'])
            symbol = data['symbol']

    result['symbol'] = symbol
    if electronic_state == 'S0':
        f_name = Path(tmp_pth, 's0_opt.com')
        geometry_2_input.geom_2_dircom(f_name, coord, symbol, keywords)
        logging.info(f'Generated input file for S0 optimization: {f_name}')
        # 调用高斯计算
        subprocess.run(['g16', 's0_opt.com'], check=True)
        logging.info(f'Completed S0 optimization calculation for {f_name}')
        subprocess.run(['formchk', 's0_opt.chk'], check=True)
        # 读取优化后的结构、能量
        with open(Path(tmp_pth, 's0_opt.fchk'), 'r') as f:
            s0_e = fchic.deck_load(f, "SCF Energy")
            coord_s0 = np.array(fchic.deck_load(f, "Current cartesian coordinates")).reshape((-1,3)) * 0.529177249
            n_occu_orbital = fchic.deck_load(f, "Number of electrons")[0] // 2
            e_orbital = fchic.deck_load(f, "Alpha MO coefficients")
            homo = e_orbital[n_occu_orbital - 1]; lumo = e_orbital[n_occu_orbital]

        result['xtbopt_coord'] = coord
        result['e_s0_s0'] = s0_e  # 保存s0态的基态的能量 
        result['s0opt_coord'] = coord_s0
        result['homo'] = homo; result['lumo'] = lumo; result['virtual_freq'] = True 
        with open(Path(tmp_pth, 's0_opt.log'), 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip().split()
            if len(line) > 2 and line[0] == 'Normal' and line[1] == 'termination':
                result['converge'] = True
            else:
                result['converge'] = False
                result['Error'].append('S0 optimization did not converge')
            for line in lines:
                if line.startswith(' Frequencies --'):
                    imaginary_freq = True if float(line.strip().split()[2]) <= 0. else False
                    result['virtual_freq'] = imaginary_freq

    elif electronic_state == 'S1':
        f_name = Path(tmp_pth, 's1_opt.com')
        geometry_2_input.geom_2_dircom(f_name, coord, symbol, keywords)
        logging.info(f'Generated input file for S1 optimization: {f_name}')
        # 调用高斯计算
        subprocess.run(['g16', 's1_opt.com'], check=True)
        logging.info(f'Completed S1 optimization calculation for {f_name}')
        subprocess.run(['formchk', 's1_opt.chk'], check=True)
        with open(Path(tmp_pth, 's1_opt.fchk'), 'r') as f:
            s1_e = fchic.deck_load(f, "Total Energy")
            coord_s1 = np.array(fchic.deck_load(f, "Current cartesian coordinates")).reshape((-1,3)) * 0.529177249
        result['e_s1_td'] = s1_e  # 保存s1态的基态的能量
        result['s1opt_coord'] = coord_s1
        result['virtual_freq'] = True
        with open(Path(tmp_pth, 's1_opt.log'), 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip().split()
            if len(line) > 2 and line[0] == 'Normal' and line[1] == 'termination':
                result['converge'] = True
            else:
                result['converge'] = False
                result['Error'].append('S1 optimization did not converge')
            for line in lines:
                if line.startswith(' Frequencies --'):
                    imaginary_freq = True if float(line.strip().split()[2]) <= 0. else False
                    result['virtual_freq'] = imaginary_freq

    elif electronic_state == 'T1':
        f_name = Path(tmp_pth, 't1_opt.com')
        geometry_2_input.geom_2_dircom(f_name, coord, symbol, keywords)
        logging.info(f'Generated input file for T1 optimization: {f_name}')
        # 调用高斯计算
        subprocess.run(['g16', 't1_opt.com'], check=True)
        logging.info(f'Completed T1 optimization calculation for {f_name}')
        subprocess.run(['formchk', 't1_opt.chk'], check=True)
        with open(Path(tmp_pth, 't1_opt.fchk'), 'r') as f:
            # t1 计算采用0 3 
            t1_e = fchic.deck_load(f, "SCF Energy")
            coord_t1 = np.array(fchic.deck_load(f, "Current cartesian coordinates")).reshape((-1,3)) * 0.529177249
        result['t1opt_coord'] = coord_t1
        result['e_t1_s0'] = t1_e  # 保存t1态的基态的能量; 
        result['virtual_freq'] = True
        with open(Path(tmp_pth, 't1_opt.log'), 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip().split()
            if len(line) > 2 and line[0] == 'Normal' and line[1] == 'termination':
                result['converge'] = True
            else:
                result['converge'] = False
                result['Error'].append('T1 optimization did not converge')
            for line in lines:
                if line.startswith(' Frequencies --'):
                    imaginary_freq = True if float(line.strip().split()[2]) <= 0. else False
                    result['virtual_freq'] = imaginary_freq
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    
    #with open(Path(out_pth, 'result.csv'), 'w') as f:
    #    df = pd.DataFrame(result)
    #    df.to_csv(f, index=False)
    return 

def evc_calculator(args:argparse.Namespace):
    '''
    momap evc 的计算, 依赖频率分析信息：fchk, log, 光物理性质edme, edma, 绝热能
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; nproc = args.nproc 
    with open(Path(out_pth, 'result.pkl'), 'rb') as f:
        result = pickle.load(f)
    if result['virtual_freq'] is True:
        logging.error('No virtual frequency found, please check the optimization results.')
        result['Error'].append('Virtual frequency found, can not perform evc calculation.')
        return
    if electronic_state == 's0-s1':
        # 检验虚频
        os.makedirs(Path(tmp_pth,'evc_s0s1'), exist_ok=True)
        subprocess.run(['cp', f'{tmp_pth}/s0_opt.fchk', f'{tmp_pth}/evc_s0s1/s0_opt.fchk'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/s1_opt.fchk', f'{tmp_pth}/evc_s0s1/s1_opt.fchk'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/s0_opt.log', f'{tmp_pth}/evc_s0s1/s0_opt.log'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/s1_opt.log', f'{tmp_pth}/evc_s0s1/s1_opt.log'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/nacme.log', f'{tmp_pth}/evc_s0s1/nacme.log'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/nacme.fchk', f'{tmp_pth}/evc_s0s1/nacme.fchk'], check=True)
        evc_s0s1 = ['do_evc=1', '', '&evc', ' ffreq(1) = "s0_opt.log"', ' ffreq(2) = "s1_opt.log"', ' fnacme = "nacme.log"', '/']
        with open(Path(tmp_pth, 'evc_s0s1', 'momap.inp'), 'w') as f:
            f.write('%s' % '\n'.join(evc_s0s1))
            f.write('\n')
        subprocess.run(['momap', '-i', 'momap.inp', '-n', str(nproc)], cwd=Path(tmp_pth, 'evc_s0s1'), check=True)

    elif electronic_state == 's0-t1':
        os.makedirs(Path(tmp_pth,'evc_s0t1'), exist_ok=True)
        subprocess.run(['cp', f'{tmp_pth}/s0_opt.fchk', f'{tmp_pth}/evc_s0t1/s0_opt.fchk'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/t1_opt.fchk', f'{tmp_pth}/evc_s0t1/t1_opt.fchk'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/s0_opt.log', f'{tmp_pth}/evc_s0t1/s0_opt.log'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/t1_opt.log', f'{tmp_pth}/evc_s0t1/t1_opt.log'], check=True)
        evc_s0t1 = ['do_evc=1', '', '&evc', ' ffreq(1) = "s0_opt.log"', ' ffreq(2) = "t1_opt.log"', '/']
        with open(Path(tmp_pth, 'evc_s0t1', 'momap.inp'), 'w') as f:
            f.write('%s' % '\n'.join(evc_s0t1))
            f.write('\n')
        subprocess.run(['momap', '-i', 'momap.inp', '-n', str(nproc)], cwd=Path(tmp_pth, 'evc_s0t1'), check=True)
        
    elif electronic_state == 's1-t1':
        os.makedirs(Path(tmp_pth,'evc_s1t1'), exist_ok=True)
        subprocess.run(['cp', f'{tmp_pth}/s1_opt.fchk', f'{tmp_pth}/evc_s1t1/s1_opt.fchk'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/t1_opt.fchk', f'{tmp_pth}/evc_s1t1/t1_opt.fchk'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/s1_opt.log', f'{tmp_pth}/evc_s1t1/s1_opt.log'], check=True)
        subprocess.run(['cp', f'{tmp_pth}/t1_opt.log', f'{tmp_pth}/evc_s1t1/t1_opt.log'], check=True)
        evc_s1t1 = ['do_evc=1', '', '&evc', ' ffreq(1) = "s1_opt.log"', ' ffreq(2) = "t1_opt.log"', '/']
        with open(Path(tmp_pth, 'evc_s1t1', 'momap.inp'), 'w') as f:
            f.write('%s' % '\n'.join(evc_s1t1))
            f.write('\n')
        subprocess.run(['momap', '-i', 'momap.inp', '-n', str(nproc)], cwd=Path(tmp_pth, 'evc_s1t1'), check=True)
    return 

def kr_calculator(args:argparse.Namespace):
    '''
    kr_calculator: 计算辐射跃迁率和非辐射跃迁率
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; nproc = args.nproc

    with open(Path(in_pth, 'result.pkl'), 'rb') as f:
        # 包含s0,s1,t1的优化结构能量&基态/激发态能量
        result = pickle.load(f)
    if result['virtual_freq'] is True:
        logging.error('No virtual frequency found, please check the optimization results.')
        result['Error'].append('Virtual frequency found, can not perform kr calculation.')
        return
    # 生成kr计算需要的物理量, s1t1能量直接用高斯计算近似，否则spin-flip失败率太高
    edme = result['edme']; edma = result['edma'] 
    if electronic_state == 's0-s1':
        e_s0 = result['e_s0_s0']; e_s1 = result['e_s1_td']
        E_s0s1 = e_s1[0] - e_s0[0]  # s0-s1的能量差
        kr_com = ['do_spec_tvcf_ft=1', 'do_spec_tvcf_spec=1', '&spec_tvcf', 'DUSHIN = .f.', 'Temp = 300 K', \
      'tmax = 7500 fs', 'dt = 0.025 fs', 'debug = .t.', 'isgauss = .f.', 'BroadenType = "lorentzian"', \
      'GFile = "ic.tvcf.gauss.dat"', 'BroadenFunc = "time"', f'Ead = {E_s0s1} au', \
      f'EDMA = {edma} debye', f'EDME = {edme} debye', 'FreqScale = 1.0', 'DSFile = "evc.dint.dat"',\
      'Emax = 0.3 au','dE = 0.00001 au','logFile = "spec.tvcf.log"','FtFile = "spec.tvcf.ft.dat"',\
      'FoFile = "spec.tvcf.fo.dat"','FoSFile = "spec.tvcf.spec.dat"','/']
        knr_com = ['do_ic_tvcf_ft = 1', 'do_ic_tvcf_spec = 1', '&ic_tvcf', 'DUSHIN = .f.', 'Temp = 300 K',\
       'tmax = 7500 fs', 'isgauss = .t.', 'BroadenType = "lorentzian"', 'GFile = "ic.tvcf.gauss.dat"',\
       'FWHM = 10 cm-1','BroadenFunc = "time"','dt = 0.025 fs', 'debug = .t.',f'Ead = {E_s0s1} au', \
       'DSFile = "evc.dint.dat"', "CoulFile = evc.cart.nac", 'Emax = 0.3 au','logFile = "ic.tvcf.log"',\
       'FtFile = "ic.tvcf.ft.dat"','FoFile = "ic.tvcf.fo.dat"','/']
        # 运行kr 
        try:
            os.makedirs(Path(tmp_pth, 'kr'), exist_ok=True)
            with open(Path(tmp_pth, 'kr', 'momap.inp'), 'w') as f:
                f.write('%s' % '\n'.join(kr_com))
                f.write('\n')
            os.system('cp evc_s0s1/evc.dint.dat kr/') if os.path.exists(Path(tmp_pth, 'evc_s0s1', 'evc.dint.dat')) \
                else os.system('cp evc_s0s1/evc.cart.dat kr/evc.dint.dat')
            os.system('cd kr && momap -i momap.inp -n %d' % nproc)
            # 运行knr 
            os.makedirs(Path(tmp_pth, 'knr'), exist_ok=True)
            with open(Path(tmp_pth, 'knr', 'momap.inp'), 'w') as f:
                f.write('%s' % '\n'.join(knr_com))
                f.write('\n')
            os.system('cp evc_s0s1/evc.dint.dat knr/') if os.path.exists(Path(tmp_pth, 'evc_s0s1', 'evc.dint.dat')) \
                else os.system('cp evc_s0s1/evc.cart.dat knr/evc.dint.dat')
            os.system('cp evc_s0s1/evc.cart.nac knr/')
            os.system('cd knr && momap -i momap.inp -n %d' % nproc)
            # 收集光谱(1nm左右采集一个,200-1000nm范围, FC_abs, FC_emi为强度)，速率常数
            wavelength = []; abs_strength = []; emi_strength = []
            with open(Path(tmp_pth, 'kr', 'spec.tvcf.spec.dat'), 'r') as f:
                lines = f.readlines()
                for line in lines[2:]:
                    line = line.strip().split()
                    if len(line) == 7:
                        if float(line[3]) > 200 and float(line[3]) < 1000:
                            wavelength.append(float(line[3]))
                            abs_strength.append(float(line[4]))
                            emi_strength.append(float(line[5]))
            result['wavelength'] = np.array(wavelength,dtype=np.float32)[::10]
            result['abs_strength'] = np.array(abs_strength,dtype=np.float32)[::10]
            result['emi_strength'] = np.array(emi_strength,dtype=np.float32)[::10]
            # 收集kr & knr 
            with open(Path(tmp_pth, 'kr', 'spec.tvcf.log'), 'r') as f:
                for line in f:
                    if line.startswith('radiative rate'):
                        kr =  float(line.strip().split()[4])
            result['kr'] = kr 
            with open(Path(tmp_pth, 'knr', 'ic.tvcf.log'), 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if line.startswith(' #1Energy(Hartree)'):
                        knr = float(lines[idx+1].strip().split()[-3])
            result['knr'] = knr 
        except:
            result['Error'].append('kr calculation did not converge')
    elif electronic_state == 's0-t1':
        e_s0 = result['e_s0_s0']; e_t1 = result['e_t1_s0']
        E_s0t1 = e_t1[0] - e_s0[0]; soc_s0t1 = result['soc1']
        knrt_com = ['do_isc_tvcf_ft=1','do_isc_tvcf_spec=1','&isc_tvcf','DUSHIN = .f.','Temp = 298 K',\
        'tmax = 7500 fs','debug = .t.','isgauss = .t.','BroadenType = "lorentzian"',\
        'GFile = "ic.tvcf.gauss.dat"','FWHM = 10 cm-1','BroadenFunc = "time"','dt = 0.025 fs',\
        f'Ead = {E_s0t1} au', f'Hso = {soc_s0t1} cm-1', 'DSFile = "evc.dint.dat"',\
        'Emax = 0.3 au','logFile = "isc.tvcf.log"',\
        'FtFile = "isc.tvcf.ft.dat"','FoFile = "isc.tvcf.fo.dat"','/']
        try:
            # 运行knrt (三重态)
            os.makedirs(Path(tmp_pth, 'knrt'), exist_ok=True)
            with open(Path(tmp_pth, 'knrt', 'momap.inp'), 'w') as f:
                f.write('%s' % '\n'.join(knrt_com))
                f.write('\n')
            os.system('cp evc_s0t1/evc.dint.dat knrt/') if os.path.exists(Path(tmp_pth, 'evc_s0t1', 'evc.dint.dat')) \
                else os.system('cp evc_s0t1/evc.cart.dat knrt/evc.dint.dat')
            os.system('cd knrt && momap -i momap.inp -n %d' % nproc)
            with open(Path(tmp_pth, 'knrt', 'isc.tvcf.log'), 'r') as f:
                lines = f.readlines()
                for idx,line in enumerate(lines):
                    if line.startswith('#         Intersystem crossing Ead'):
                        line = line.strip().split()
                        knrt = float(line[-6])
            result['knrt'] = knrt
        except:
            result['Error'].append('knrt calculation did not converge')
    elif electronic_state == 's1-t1':
        e_s1 = result['e_s1_td']; e_t1 = result['e_t1_s0']
        E_s1t1 = e_t1[0] - e_s1[0]; soc_s1t1 = result['soc0']     
        kisc_com = ['do_isc_tvcf_ft=1','do_isc_tvcf_spec=1','&isc_tvcf','DUSHIN = .f.','Temp = 298 K',\
        'tmax = 7500 fs','debug = .t.','isgauss = .t.','BroadenType = "lorentzian"',\
        'GFile = "ic.tvcf.gauss.dat"','FWHM = 10 cm-1','BroadenFunc = "time"','dt = 0.025 fs',\
        f'Ead = {E_s1t1} au', f'Hso = {soc_s1t1} cm-1','DSFile = "evc.dint.dat"',\
        'Emax = 0.3 au','logFile   = "isc.tvcf.log"','FtFile    = "isc.tvcf.ft.dat"',\
        'FoFile = "isc.tvcf.fo.dat"','/']

        try:
            # 运行kisc (单重态)
            os.makedirs(Path(tmp_pth, 'kisc'), exist_ok=True)
            with open(Path(tmp_pth, 'kisc', 'momap.inp'), 'w') as f:
                f.write('%s' % '\n'.join(kisc_com))
                f.write('\n')
            os.system('cp evc_s1t1/evc.dint.dat kisc/') if os.path.exists(Path(tmp_pth, 'evc_s1t1', 'evc.dint.dat')) \
                else os.system('cp evc_s1t1/evc.cart.dat kisc/evc.dint.dat')
            os.system('cd kisc && momap -i momap.inp -n %d' % nproc)
            with open(Path(tmp_pth, 'kisc', 'isc.tvcf.log'), 'r') as f:
                lines = f.readlines()
                for idx,line in enumerate(lines):
                    if line.startswith('# Reverse Intersystem'):
                        line0 = lines[idx-1].strip().split()
                        line1 = lines[idx].strip().split()
                        kisc = float(line0[-6])
                        krisc = float(line1[-6])
                        result['kisc'] = kisc; result['krisc'] = krisc
        except:
            result['Error'].append('kisc calculation did not converge')
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 
    
def qchem_single_calculator(args:argparse.Namespace):
    '''
    qchem的单点性质计算, 用于计算s态与t态的耦合, delta Est存在自旋污染，我们不考虑了
    '''
    gau_2_qchem = {'3-21g':'def2-svp', 'def2svp': 'def2-svp', 'def2tzvp': 'def2-tzvp', 'def2qzvp': 'def2-qzvp', 'EmpiricalDispersion=GD3BJ b3lyp': 'b3lyp'}
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    nproc = args.nproc; memory = args.memory; method = gau_2_qchem[args.method]; basis = gau_2_qchem[args.basis]
    qchem = args.qchem

    keywords = ['$end','','$rem', 'jobtype sp', f'basis {basis}', f'method {method}', \
                'cis_n_roots 4', 'rpa 2', 'calc_soc true', 'dft_d d3', \
                'sym_ignore true', 'symmetry off', '$end']
    with open(Path(in_pth, 'result.pkl'), 'rb') as f:
        result = pickle.load(f)
    symbol = result['symbol']
    coord = result['s0opt_coord']
    f_name = Path(tmp_pth, 's0_sf.in')
    geometry_2_input.geom_2_dirin(f_name, coord, symbol, keywords)
    subprocess.run([qchem, '-nt', str(nproc), 's0_sf.in', 's0_sf.out'], check=True)
    # 读取soc
    with open(Path(tmp_pth, 's0_sf.out'), 'r') as f:
        lines = f.readlines()
        try:
            for idx, line in enumerate(lines):
                if line.startswith('Total SOC between the S1 state and excited triplet'):
                    soc0 = lines[idx+1].strip().split()[-2]
                    soc0 = float(soc0)
                elif line.startswith('Total SOC between the singlet ground state and excited triplet'):
                    soc1 = lines[idx+1].strip().split()[-2]
                    soc1 = float(soc1)
            result['soc0'] = soc0
            result['soc1'] = soc1
        except Exception as e:
            result['Error'].append(f'soc calculation did not converge: {e}')
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return

def orca_single_calculator():
    '''
    orca软件的单点性质计算，暂时未用到
    '''
    return 

def gaussian_single_calculator(args:argparse.Namespace):
    '''
    gaussian 16的单点性质计算, 需要补一个nacme的计算
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    nproc = args.nproc; memory = args.memory; method = args.method; basis = args.basis
    charge = args.charge; multiplicity = args.multiplicity
    electronic_state = args.electronic_state
    if os.path.exists(f'{out_pth}/result.pkl'):
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}
    # s0_td / s1_td 计算 
    keywords = [f'%chk={electronic_state.lower()}_td.chk',f'%nproc={nproc}', f'%mem={memory}', \
                   f'# td {method}/{basis}', '', 'td', '', f'{charge} {multiplicity}']
    if electronic_state == 'S0':
        f_name = Path(tmp_pth, 's0_td.com')
        coord = result['s0opt_coord'] 
        symbol = result['symbol']
        geometry_2_input.geom_2_dircom(f_name, coord, symbol, keywords)
        subprocess.run(['g16', 's0_td.com'], check=True)
        logging.info(f'Completed S0 TD calculation for {f_name}')
        subprocess.run(['formchk', 's0_td.chk'], check=True)
        with open(Path(tmp_pth, 's0_td.fchk'), 'r') as f:
            e_s0_s0 = fchic.deck_load(f, "SCF Energy")
            e_s0_td = fchic.deck_load(f, "Total Energy")
        result['e_s0_s0'] = e_s0_s0  # 保存s0态的基态的能量
        result['e_s0_td'] = e_s0_td
        with open(Path(tmp_pth, 's0_td.log'), 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip().split()
            if len(line) > 2 and line[0] == 'Normal' and line[1] == 'termination':
                result['converge'] = True
            else:
                result['converge'] = False
                result['Error'].append('S0 TD calculation did not converge')
            for idx, line in enumerate(lines):
                if line.startswith(' Ground to excited state transition densities written to RWF'):
                    edma = lines[idx + 3].strip().split()[-2]
                    edma = float(edma)**0.5 * 2.5423
            result['edma'] = edma

    elif electronic_state == 'S1':
        f_name = Path(tmp_pth, 's1_td.com')
        coord = result['s1opt_coord']
        symbol = result['symbol']
        geometry_2_input.geom_2_dircom(f_name, coord, symbol, keywords)
        subprocess.run(['g16', 's1_td.com'], check=True)
        logging.info(f'Completed S1 TD calculation for {f_name}')
        subprocess.run(['formchk', 's1_td.chk'], check=True)
        with open(Path(tmp_pth, 's1_td.fchk'), 'r') as f:
            e_s1_td = fchic.deck_load(f, "Total Energy")
            e_s1_s0 = fchic.deck_load(f, "SCF Energy")
        result['e_s1_td'] = e_s1_td
        result['e_s1_s0'] = e_s1_s0
        with open(Path(tmp_pth, 's1_td.log'), 'r') as f:
            lines = f.readlines()
            line = lines[-1].strip().split()
                        
            if len(line) > 2 and line[0] == 'Normal' and line[1] == 'termination':
                result['converge'] = True
            else:
                result['converge'] = False
                result['Error'].append('S1 TD calculation did not converge')
            for idx, line in enumerate(lines):
                if line.startswith(' Ground to excited state transition densities written to RWF'):
                    edme = lines[idx + 3].strip().split()[-2]
                    edme = float(edme)**0.5 * 2.5423
            result['edme'] = edme

    elif electronic_state == 'nacme':
        # nacme 计算
        f_name = Path(tmp_pth, 'nacme.com')
        coord_s0 = result['s0opt_coord']; symbol = result['symbol']
        keywords = [f'%chk=nacme.chk',f'%nproc={nproc}', f'%mem={memory}', \
                   f'# p td {method}/{basis} prop=(fitcharge,field) iop(6/22=-4,6/29=1,6/30=0,6/17=2) nosymm', '', 'td', '', f'{charge} {multiplicity}']
        geometry_2_input.geom_2_dircom(f_name, coord_s0, symbol, keywords)
        subprocess.run(['g16', 'nacme.com'], check=True)
        f_name = Path(tmp_pth, 's0_opt.com')
        keywords = [f'%chk=s0_opt.chk',f'%nproc={nproc}', f'%mem={memory}', \
                     f'# {method}/{basis} freq nosymm', '', 'opt', '', f'{charge} {multiplicity}']
        coord_s0 = result['s0opt_coord']
        geometry_2_input.geom_2_dircom(f_name, coord_s0, symbol, keywords)
        subprocess.run(['g16', 's0_opt.com'], check=True)
        logging.info(f'Completed NACME calculation for {f_name}')
        subprocess.run(['formchk', 'nacme.chk'], check=True)
        subprocess.run(['formchk', 's0_opt.chk'], check=True)
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 

def gen_property_slurm(args:argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); g16_env = args.g16_env; platform_env = args.platform_env; qchem_env = args.qchem_env
    script_pth = Path(args.script_pth); nproc = args.nproc; memory = args.memory
    method = args.method; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; q_chem = args.q_chem
    
    calc_files = glob(f'{in_pth}/*/input.com')
    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(g16_env.strip().split(';')) if g16_env else []
        slurm_txt.extend(qchem_env.strip().split(';')) if qchem_env else []
        # 计算工作流调用
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}',f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S0', 
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S1',
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state T1',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state S0',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state S1',
                      f'python {script_pth} qchem_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --qchem "{q_chem}"'])   
        slurm_txt.extend([f'rm -rf {tmp_pth}/mol_{idx}']) # 清理临时文件夹    
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def gen_spectrum_slurm(args:argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); g16_env = args.g16_env; platform_env = args.platform_env; qchem_env = args.qchem_env
    script_pth = Path(args.script_pth); nproc = args.nproc; memory = args.memory
    method = args.method; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; q_chem = args.q_chem; momap_env = args.momap_env
    calc_files = glob(f'{in_pth}/*/input.com')
    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(g16_env.strip().split(';')) if g16_env else []
        slurm_txt.extend(qchem_env.strip().split(';')) if qchem_env else []

        # 计算工作流调用
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}',f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S0', 
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S1',
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state T1',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state S0',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state S1',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state nacme',
                      f'python {script_pth} qchem_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --qchem "{q_chem}"'])  
        slurm_txt.extend(momap_env.strip().split(';')) if momap_env else []
        # 如果只需要计算光谱，evcs0s1 & krs0s1既可
        slurm_txt.extend([f'python {script_pth} evc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s0-s1',
                      f'python {script_pth} kr_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s0-s1'])
        slurm_txt.extend([f'rm -rf {tmp_pth}/mol_{idx}'])  # 清理临时文件夹
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def gen_plqy_slurm(args:argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); g16_env = args.g16_env; platform_env = args.platform_env; qchem_env = args.qchem_env
    script_pth = Path(args.script_pth); nproc = args.nproc; memory = args.memory
    method = args.method; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; q_chem = args.q_chem; momap_env = args.momap_env
    calc_files = glob(f'{in_pth}/*/input.com')
    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(g16_env.strip().split(';')) if g16_env else []
        slurm_txt.extend(qchem_env.strip().split(';')) if qchem_env else []
        
        # 计算工作流调用
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}',f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S0', 
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S1',
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state T1',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state S0',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state S1',
                      f'python {script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state nacme',
                      f'python {script_pth} qchem_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {memory} --method "{method}" --basis "{basis}" --qchem "{q_chem}"'])
        slurm_txt.extend(momap_env.strip().split(';')) if momap_env else []
        # 如果只需要计算光谱，evcs0s1 & krs0s1既可
        slurm_txt.extend([f'python {script_pth} evc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s0-s1',
                      f'python {script_pth} evc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s1-t1',
                      f'python {script_pth} evc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s0-t1',
                      f'python {script_pth} kr_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s0-s1',
                      f'python {script_pth} kr_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s1-t1',
                      f'python {script_pth} kr_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --electronic_state s0-t1'])
        slurm_txt.extend([f'rm -rf {tmp_pth}/mol_{idx}']) # 清理临时文件夹
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def run_slurm(args:argparse.Namespace):
    slurm_task_path = Path(args.slurm_task_pth)
    for slurm_file in slurm_task_path.glob('*.slurm'):
        os.system(f'cd {slurm_task_path} && sbatch {slurm_file}')
    os.system(f'rm -rf {slurm_task_path}')  # 清理slurm文件夹
    return 

def collect_data(args:argparse.Namespace):
    in_pth = Path(args.in_pth); task_type = args.task_type; out_pth = Path(args.out_pth)
    # 保存为lmdb 
    f_files = glob(f'{out_pth}/*/result.pkl')
    results = []
    for f_name in f_files:
        try:
            with open(f_name, 'rb') as f:
                key = os.path.basename(os.path.dirname(f_name))
                df = pickle.load(f)
                error = df['Error'] if 'Error' in df.keys() else []
                if len(error) == 0:
                    if task_type == 'plqy':
                        kr = df['kr']; knr = df['knr']; kisc = df['kisc']
                        krisc = df['krisc']; knrt = df['knrt']
                        plqy_pf = kr / (kisc + kr + knr); plqy_isc = kisc / (kisc + kr + knr)
                        plqy_risc = krisc / (krisc + knrt); plqy_df = (plqy_isc * plqy_risc / (1 - plqy_isc * plqy_risc)) * plqy_pf
                        plqy = plqy_df + plqy_pf
                        df['plqy'] = plqy
                        lifetime_kr = 1/kr if kr != 0 else 0
                        df['lifetime_kr'] = lifetime_kr
                        # 或许需要寿命计算，速率常数的倒数
                results.append((key, df))
        except Exception as e:
            logging.error(f'Error processing {f_name}: {e}')
            continue
        #    continue
    # 保存为pickle 文件
    outputfilename = str(Path(out_pth, 'results.lmdb'))
    env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(1024*1024*1024*50),
        )
    txn_write = env_new.begin(write=True)
    for idx, res in enumerate(results):
        key, df = res
        txn_write.put(key.encode("ascii"), pickle.dumps(df, protocol=-1))
        if idx % 100 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)
    txn_write.commit()
    env_new.close()
    return 


def main():
    parser = argparse.ArgumentParser(description='Run different photophysical properties calculation tasks')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # subparser for gen input file, 支持csv文件
    parser_gen_input = subparsers.add_parser('gen_input', help='Generate xyz')
    parser_gen_input.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for xyz or csv file')
    
    # subparser for structure optimization
    parser_structure_opt = subparsers.add_parser('structure_opt', help='Run structure optimization')
    parser_structure_opt.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_structure_opt.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')
    parser_structure_opt.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_structure_opt.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    parser_structure_opt.add_argument('--memory', type=str, default='10GB', help='Memory in GB to use for optimization')
    parser_structure_opt.add_argument('--method', type=str, default='b3lyp EmpiricalDispersion=GD3BJ', help='Method for optimization')
    parser_structure_opt.add_argument('--basis', type=str, default='def2svp', help='Basis set for optimization')
    parser_structure_opt.add_argument('--charge', type=int, default=0, help='Charge of the system')
    parser_structure_opt.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the system')
    parser_structure_opt.add_argument('--freq', type=bool, default=True, help='Whether to perform frequency analysis after optimization')
    parser_structure_opt.add_argument('--electronic_state', type=str, default='S0', choices=['S0', 'S1', 'T1'], help='Electronic state for optimization')
    
    # subparser for qchem single point calculation
    parser_qchem_single = subparsers.add_parser('qchem_single_calculator', help='Run single point calculation using Q-Chem')
    parser_qchem_single.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_qchem_single.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')
    parser_qchem_single.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_qchem_single.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    parser_qchem_single.add_argument('--memory', type=str, default='10GB', help='Memory in GB to use for optimization')
    parser_qchem_single.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_qchem_single.add_argument('--basis', type=str, default='def2-svp', help='Basis set for optimization')
    parser_qchem_single.add_argument('--qchem', type=str, default='/public/home/chengz/soft/Q-Chem/bin/qchem', help='qchem path')
    
    # subparser for gaussian single point calculation
    parser_gau_single = subparsers.add_parser('gau_single_calculator', help='Run single point calculation using Gaussian')
    parser_gau_single.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_gau_single.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')   
    parser_gau_single.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_gau_single.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    parser_gau_single.add_argument('--memory', type=str, default='10GB', help='Memory in GB to use for optimization')
    parser_gau_single.add_argument('--method', type=str, default='b3lyp EmpiricalDispersion=GD3BJ', help='Method for optimization')
    parser_gau_single.add_argument('--basis', type=str, default='def2svp', help='Basis set for optimization')
    parser_gau_single.add_argument('--charge', type=int, default=0, help='Charge of the system')
    parser_gau_single.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the system')
    parser_gau_single.add_argument('--electronic_state', type=str, default='S0', choices=['S0', 'S1', 'T1', 'nacme'], help='Electronic state for optimization')

    # subparser for evc calculation
    parser_evc_calculator = subparsers.add_parser('evc_calculator', help='Run EVC calculation')
    parser_evc_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_evc_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results') 
    parser_evc_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_evc_calculator.add_argument('--electronic_state', type=str, default='s0-s1', choices=['s0-s1', 's0-t1', 's1-t1'], help='Electronic state for EVC calculation')
    parser_evc_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    # subparser for kr calculation
    parser_kr_calculator = subparsers.add_parser('kr_calculator', help='Run KR calculation')
    parser_kr_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_kr_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')
    parser_kr_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_kr_calculator.add_argument('--electronic_state', type=str, default='s0-s1', choices=['s0-s1', 's0-t1', 's1-t1'], help='Electronic state for KR calculation')
    parser_kr_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for generating property slurm
    parser_gen_property_slurm = subparsers.add_parser('gen_property_slurm', help='Generate property calculation slurm script')
    parser_gen_property_slurm.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_gen_property_slurm.add_argument('--slurm_task_pth', type=str, default='/public/home/chengz/FunMG/task', help='Input path for slurm')
    parser_gen_property_slurm.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for software calculation')
    parser_gen_property_slurm.add_argument('--g16_env', type=str,  default='export GAUSS_SCRDIR=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Environment for Gaussian 16')
    parser_gen_property_slurm.add_argument('--qchem_env', type=str, default='export QC=/public/home/chengz/soft/Q-Chem;export QCAUX=/public/home/chengz/soft/Q-Chem/qcaux;export QCSCRATCH=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Q-Chem path')
    parser_gen_property_slurm.add_argument('--platform_env', type=str, default='#SBATCH -p kshcnormal', help='Environment for Sugon')
    parser_gen_property_slurm.add_argument('--script_pth', type=str, default='/public/home/chengz/FunMG/job_adv.py', help='Path for job_adv.py')
    parser_gen_property_slurm.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    parser_gen_property_slurm.add_argument('--memory', type=str, default='10GB', help='Memory in GB to use for optimization')
    parser_gen_property_slurm.add_argument('--method', type=str, default='b3lyp EmpiricalDispersion=GD3BJ', help='Method for optimization')
    parser_gen_property_slurm.add_argument('--basis', type=str, default='def2svp', help='Basis set for optimization')
    parser_gen_property_slurm.add_argument('--charge', type=int, default=0, help='Charge of the system')
    parser_gen_property_slurm.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the system')
    parser_gen_property_slurm.add_argument('--freq', type=bool, default=True, help='Whether to perform frequency analysis after optimization')
    parser_gen_property_slurm.add_argument('--q_chem', type=str, default='/public/home/chengz/soft/Q-Chem/bin/qchem', help='Q-Chem path') 

    # subparser for generating spectrum slurm
    parser_gen_spec_slurm = subparsers.add_parser('gen_spec_slurm', help='Generate spectrum calculation slurm script')
    parser_gen_spec_slurm.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_gen_spec_slurm.add_argument('--slurm_task_pth', type=str, default='/public/home/chengz/FunMG/task', help='Input path for slurm')
    parser_gen_spec_slurm.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for software calculation')
    parser_gen_spec_slurm.add_argument('--g16_env', type=str, default=['export GAUSS_SCRDIR=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID'], help='Environment for Gaussian 16')
    parser_gen_spec_slurm.add_argument('--qchem_env', type=str, default=['export QC=/public/home/chengz/soft/Q-Chem', 'export QCAUX=/public/home/chengz/soft/Q-Chem/qcaux','export QCSCRATCH=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/'], help='Q-Chem path')
    parser_gen_spec_slurm.add_argument('--platform_env', type=str, default=['#SBATCH -p kshcnormal'], help='Environment for Sugon')
    parser_gen_spec_slurm.add_argument('--script_pth', type=str, default='/public/home/chengz/FunMG/job_adv.py', help='Path for job_adv.py')
    parser_gen_spec_slurm.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    parser_gen_spec_slurm.add_argument('--memory', type=str, default='10GB', help='Memory in GB to use for optimization')
    parser_gen_spec_slurm.add_argument('--method', type=str, default='b3lyp EmpiricalDispersion=GD3BJ', help='Method for optimization')
    parser_gen_spec_slurm.add_argument('--basis', type=str, default='def2svp', help='Basis set for optimization')
    parser_gen_spec_slurm.add_argument('--charge', type=int, default=0, help='Charge of the system')
    parser_gen_spec_slurm.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the system')
    parser_gen_spec_slurm.add_argument('--freq', type=bool, default=True, help='Whether to perform frequency analysis after optimization')
    parser_gen_spec_slurm.add_argument('--q_chem', type=str, default='/public/home/chengz/soft/Q-Chem/bin/qchem', help='Q-Chem path')
    parser_gen_spec_slurm.add_argument('--momap_env', type=str, default=['source /public/home/chengz/MOMAP-2022A/env.sh'], help='MoMaP path')

    # subparser for generating plqy slurm
    parser_gen_plqy_slurm = subparsers.add_parser('gen_plqy_slurm', help='Generate spectrum calculation slurm script')
    parser_gen_plqy_slurm.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_gen_plqy_slurm.add_argument('--slurm_task_pth', type=str, default='/public/home/chengz/FunMG/task', help='Input path for slurm')
    parser_gen_plqy_slurm.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for software calculation')
    parser_gen_plqy_slurm.add_argument('--g16_env', type=str,  default=['export GAUSS_SCRDIR=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID'], help='Environment for Gaussian 16')
    parser_gen_plqy_slurm.add_argument('--qchem_env', type=str,  default=['export QC=/public/home/chengz/soft/Q-Chem', 'export QCAUX=/public/home/chengz/soft/Q-Chem/qcaux','export QCSCRATCH=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/'], help='Q-Chem path')
    parser_gen_plqy_slurm.add_argument('--platform_env', type=str,  default=['#SBATCH -p kshcnormal'], help='Environment for Sugon')
    parser_gen_plqy_slurm.add_argument('--script_pth', type=str, default='/public/home/chengz/FunMG/job_adv.py', help='Path for job_adv.py')
    parser_gen_plqy_slurm.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    parser_gen_plqy_slurm.add_argument('--memory', type=str, default='10GB', help='Memory in GB to use for optimization')
    parser_gen_plqy_slurm.add_argument('--method', type=str, default='b3lyp EmpiricalDispersion=GD3BJ', help='Method for optimization')
    parser_gen_plqy_slurm.add_argument('--basis', type=str, default='def2svp', help='Basis set for optimization')
    parser_gen_plqy_slurm.add_argument('--charge', type=int, default=0, help='Charge of the system')
    parser_gen_plqy_slurm.add_argument('--multiplicity', type=int, default=1, help='Multiplicity of the system')
    parser_gen_plqy_slurm.add_argument('--freq', type=bool, default=True, help='Whether to perform frequency analysis after optimization')
    parser_gen_plqy_slurm.add_argument('--q_chem', type=str, default='/public/home/chengz/soft/Q-Chem/bin/qchem', help='Q-Chem path')
    parser_gen_plqy_slurm.add_argument('--momap_env', type=str, default=['source /public/home/chengz/MOMAP-2022A/env.sh'], help='MoMaP path')
    
    # subparser for run slurm  
    parser_run_slurm = subparsers.add_parser('run_slurm', help='Run slurm scripts')
    parser_run_slurm.add_argument('--slurm_task_pth', type=str, default='/public/home/chengz/FunMG/task', help='Path for slurm scripts')
    # subparser for collecting data
    parser_collect_data = subparsers.add_parser('collect_data', help='Collect data from calculation results')
    parser_collect_data.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for calculation results')
    parser_collect_data.add_argument('--task_type', type=str, default='property', choices=['property', 'spectrum', 'plqy'], help='Type of task to collect data for')
    parser_collect_data.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for collected data')

    args = parser.parse_args()
    ########################################## 
    # 单一任务计算
    ##########################################
    if args.command == 'gen_input':
        # 全部为生成slurm 
        gen_com(args)

    if args.command == 'structure_opt':
        # 单个opt任务
        opt_calculator(args)

    if args.command == 'qchem_single_calculator':   
        # 单个qchem单点任务 
        qchem_single_calculator(args)

    if args.command == 'gau_single_calculator':
        gaussian_single_calculator(args)

    if args.command == 'evc_calculator':
        evc_calculator(args)

    if args.command == 'kr_calculator':
        kr_calculator(args)
     
    ##################################################
    # 根据需求组装单个任务,并生成相应的slurm 脚本
    ##################################################
    if args.command == 'gen_property_slurm':
        # 生成从分子结构生成-光物理性质计算脚本; 提供软件环境slurm环境；提供计算参数
        os.makedirs(args.slurm_task_pth, exist_ok=True)  # 确保任务路径存在
        gen_property_slurm(args)

    if args.command == 'gen_spec_slurm':
        # 生成从分子结构生成-光物理性质计算-吸收发射光谱
        os.makedirs(args.slurm_task_pth, exist_ok=True)
        gen_spectrum_slurm(args)

    if args.command == 'gen_plqy_slurm':
        # 生成从分子结构生成-光物理性质计算-吸收、发射光谱-荧光量子产率脚本
        os.makedirs(args.slurm_task_pth, exist_ok=True)
        gen_plqy_slurm(args)
    

    ####################################################
    # 运行slurm 脚本
    ####################################################
    if args.command == 'run_slurm':
        run_slurm(args)
    
    ####################################################
    # 收集数据
    ####################################################
    if args.command == 'collect_data':
        # 收集数据
        collect_data(args)

        

if __name__ == '__main__':
    main()
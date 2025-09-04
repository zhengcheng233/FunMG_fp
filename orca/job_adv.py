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
from utils import geometry_2_input, smi_2_geom, slurm, orca_2_gau
from glob import glob
import lmdb 
import pickle 
import traceback

def read_data(args:argparse.Namespace):
    '''
    读取数据，生成xyz文件
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth)
    try:
        with open(f'{in_pth}/result.pkl', 'rb') as fp:
            results = pickle.load(fp)
        return results
    except Exception as e:
        print(f"read_data: {in_pth}/result.pkl not found, skipping {in_pth}.")
        return None 
        
def gen_data(args:argparse.Namespace):
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth)
    with open(in_pth, 'rb') as f:
        data = pickle.load(f)
    for ii in range(len(data)):
        os.makedirs(f'{out_pth}/frame_{ii}', exist_ok=True)
        with open(f'{out_pth}/frame_{ii}/result.pkl', 'wb') as f:
            pickle.dump(data[ii], f)
    return 


def spec_calculator(args:argparse.Namespace):
    # 包含磷光、荧光计算, 首先orca_2_gau 生成 hessian， 然后根据磷光/荧光 计算光谱 
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method; sov = args.sov
    freq_file_0 = args.freq_file_0; freq_file_1 = args.freq_file_1 
    nproc = args.nproc

    with open(f'{in_pth}/result.pkl', 'rb') as f:
        result = pickle.load(f)
        #e_ad = result['e_ad'] * 219474.6665 if 'e_ad' in result.keys() else None
    #if e_ad is None:
    #    return  
    hessian_s0 = result.get('hessian_S0', None); hessian_s1 = result.get('hessian_S1', None)
    hessian_t1 = result.get('hessian_T1', None)
    e_s0_min = result.get('e_s0_s0', None)[0] if 'e_s0_s0' in result.keys() else None 
    e_s1_min = result.get('e_s1_td', None)[0] if 'e_s1_td' in result.keys() else None 
    e_t1_min = result.get('e_t1_td', None)[0] if 'e_t1_td' in result.keys() else None

    if 'e_s0_min' in result.keys():
        e_s0_min = result.get('e_s0_min', None) if 'e_s0_min' in result.keys() else None 
        e_s1_min = result.get('e_s1_min', None) if 'e_s1_min' in result.keys() else None 
        e_t1_min = result.get('e_t1_min', None) if 'e_t1_min' in result.keys() else None
    do_ht = False
    if electronic_state == 'Abs' and hessian_s0 is not None and e_s0_min is not None and hessian_s1 is not None and e_s1_min is not None:
        e_ad = (e_s1_min - e_s0_min) * 219474.6665  # 转换为cm-1
        keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF ESD(ABS)', \
                    '%maxcore 3000', f'%pal nprocs {nproc} end', '%TDDFT', '  NROOTS 30', \
                    '  TDA TRUE', '  IROOT 1', 'END', f'%ESD ', 'TCUTFREQ 50', f'  GSHESSIAN "{freq_file_0}"', \
                    f'  ESHESSIAN "{freq_file_1}"', f'  DOHT {do_ht}', '  LINES VOIGT', \
                    '  LINEW 75', '  INLINEW 300', 'END', '', '* xyz 0 1', '']
        # freq s0 & s1
        orca_2_gau.hessian_convert(hessian_s0, f'{tmp_pth}/{freq_file_0}', len(result['symbol']), result['s0opt_coord'], result['symbol'], 1)
        orca_2_gau.hessian_convert(hessian_s1, f'{tmp_pth}/{freq_file_1}', len(result['symbol']), result['s1opt_coord'], result['symbol'], 1)
        with open(Path(tmp_pth, 'abs.inp'), 'w') as f:
            f.write('\n'.join(keywords))
            for ii in range(len(result['symbol'])):
                f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['s0opt_coord'][ii][2]))
            f.write('*\n')
        # 运行orca, 获取光谱文件 *.spectrum, energy 对应波数； TotalSpectrum 对应总的强度 
        # cm-1 = 10**7/nm
        spectrum_x = []; spectrum_y = []
        with open(Path(tmp_pth, 'abs_err.out'), 'w') as f1, open(Path(tmp_pth, 'abs_out.out'), 'w') as f2:
            subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'abs.inp'], check=True, stdout=f2, stderr=f1)
        with open(Path(tmp_pth, 'abs.spectrum'), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.strip().split()
                if len(line) == 4:
                    spectrum_x.append(float(line[0])); spectrum_y.append(float(line[1]))
        result['abs_wavelength'] = 10**7 / np.array(spectrum_x); result['abs_fosc'] = np.array(spectrum_y)
            
    elif electronic_state == 'Abs_base':
        keywords = [f'!{func} d3bj def2-svp def2/J RIJCOSX tightSCF', \
                    '%maxcore 3000', f'%pal nprocs {nproc} end', '%TDDFT', '  NROOTS 30', \
                    'TDA TRUE', '  IROOT 1', 'END', '*xyz 0 1', '']
        with open(Path(tmp_pth, 'abs.inp'), 'w') as f:
            f.write('\n'.join(keywords))
            for ii in range(len(result['symbol'])):
                f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['s0opt_coord'][ii][2]))
            f.write('*\n')
        # 运行orca, 获取光谱文件 *.spectrum, energy 对应波数； TotalSpectrum 对应总的强度 
        # cm-1 = 10**7/nm
        spectrum_x = []; spectrum_y = []
        with open(Path(tmp_pth, 'abs_err.out'), 'w') as f1, open(Path(tmp_pth, 'abs_out.out'), 'w') as f2:
            subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'abs.inp'], check=True, stdout=f2, stderr=f1)
       
        # 需要保留跃迁偶极矩/振子强度、吸收能； 需要读取K*K 
        K2 = None; abs_energy = []; abs_fosc = []
        with open(Path(tmp_pth,'abs_out.out'),'r') as f:
            lines = f.readlines(); read = False
            for line in lines:
                line = line.strip().split()
                if 'K*K' in line:
                    K2 = 31 if line[0] == 'Error' else float(line[-1])
                if read == True:
                    if len(line) == 11:
                        abs_energy.append(float(line[5]))
                        abs_fosc.append(float(line[6]))
                if len(line) > 5 and line[0] == 'ABSORPTION' and line[1] == 'SPECTRUM' and line[4] == 'ELECTRIC':
                    read = True 
                if len(line) > 5 and line[0] == 'ABSORPTION' and line[1] == 'SPECTRUM' and line[4] == 'VELOCITY':
                    read = False
        result['abs_energy'] = np.array(abs_energy) if len(abs_energy) > 0 else None
        result['abs_fosc'] = np.array(abs_fosc) if len(abs_fosc) > 0 else None
        result['K2'] = K2

    elif electronic_state == 'Emi_fluor' and hessian_s0 is not None and e_s0_min is not None and hessian_s1 is not None and e_s1_min is not None:
        e_ad = (e_s1_min - e_s0_min) * 219474.6665

        keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF ESD(FLUOR)', \
                    '%maxcore 3000', f'%pal nprocs {nproc} end', '%TDDFT', '  NROOTS 10', \
                    '  TDA TRUE', '  IROOT 1', 'END', f'%ESD ', 'TCUTFREQ 50', f'  GSHESSIAN "{freq_file_0}"', \
                    f'  ESHESSIAN "{freq_file_1}"', f'  DOHT {do_ht}', '  LINES VOIGT', \
                    '  LINEW 75', '  INLINEW 300', 'END', '', '* xyz 0 1', '']
        # freq s0 & s1 
        orca_2_gau.hessian_convert(hessian_s0, f'{tmp_pth}/{freq_file_0}', len(result['symbol']), result['s0opt_coord'], result['symbol'])
        orca_2_gau.hessian_convert(hessian_s1, f'{tmp_pth}/{freq_file_1}', len(result['symbol']), result['s1opt_coord'], result['symbol'])
        with open(Path(tmp_pth, 'emi_fluor.inp'), 'w') as f:
            f.write('\n'.join(keywords))
            for ii in range(len(result['symbol'])):
                f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['s0opt_coord'][ii][2]))
            f.write('end\n')
        spectrum_x = []; spectrum_y = []
        with open(Path(tmp_pth, 'emi_fluor_err.out'), 'w') as f1, open(Path(tmp_pth, 'emi_fluor_out.out'), 'w') as f2:
            subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'emi_fluor.inp'], check=True, stdout=f2, stderr=f1)
        with open(Path(tmp_pth, 'emi_fluor.spectrum'), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.strip().split()
                if len(line) == 4:
                    spectrum_x.append(float(line[0])); spectrum_y.append(float(line[1]))
        result['emi_fluor_wavelength'] = 10**7 / np.array(spectrum_x); result['emi_fluor_fosc'] = np.array(spectrum_y)
        kr = None 
        with open(Path(tmp_pth, 'emi_fluor_out.out'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) > 4 and line[0] == 'The' and line[1] == 'calculated' and line[2] == 'fluorescence':
                    kr = float(line[-2])
        result['kr'] = kr if kr is not None else None
      
    elif electronic_state == 'Emi_phosphor' and hessian_s0 is not None and e_s0_min is not None and hessian_t1 is not None and e_t1_min is not None:
        e_ad = (e_t1_min - e_s0_min) * 219474.6665
        # 计算磷光光谱  需要soc， 基组改

        ss = result['symbol'][0]
        keywords = [[f'! {func} d3bj  ZORA ZORA-def2-SVP SARC/J tightSCF ESD(PHOSP) RI-SOMF(1X)', \
                    '%maxcore 3000', f'%pal nprocs {nproc} end', '%TDDFT', '  NROOTS 10', \
                    '  DOSOC TRUE', '  TDA TRUE', f'  IROOT {iroot}', 'END', f'%ESD ', 'TCUTFREQ 50', f'  GSHESSIAN "{freq_file_0}"', \
                    f'  TSHESSIAN "{freq_file_1}"', f'  DOHT {do_ht}', f'DELE {e_ad}', '  LINES VOIGT', \
                    '  LINEW 75', '  INLINEW 300', 'END', f'%basis newgto {ss} "SARC-ZORA-SVP" end', 'end', '', '* xyz 0 1'] for iroot in [1, 2, 3]]   
        # freq s0 & t1
        orca_2_gau.hessian_convert(hessian_s0, f'{tmp_pth}/{freq_file_0}', len(result['symbol']), result['s0opt_coord'], result['symbol'], 1)
        orca_2_gau.hessian_convert(hessian_t1, f'{tmp_pth}/{freq_file_1}', len(result['symbol']), result['t1opt_coord'], result['symbol'], 3)
        with open(Path(tmp_pth, 'emi_phosphor.inp'), 'w') as f:
            f.write('\n'.join(keywords[0])); f.write('\n')
            for ii in range(len(result['symbol'])):
                f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['s0opt_coord'][ii][2]))
            f.write('end\n')
            f.write('\n $NEW_JOB \n')
            f.write('\n'.join(keywords[1])); f.write('\n')
            for ii in range(len(result['symbol'])):
                f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['t1opt_coord'][ii][2]))
            f.write('end\n')
            f.write('\n $NEW_JOB \n')
            f.write('\n'.join(keywords[2])); f.write('\n')
            for ii in range(len(result['symbol'])):
                f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['t1opt_coord'][ii][2]))
            f.write('end\n')
        spectrum_x = []; spectrum_y = []
        with open(Path(tmp_pth, 'emi_phosphor_err.out'), 'w') as f1, open(Path(tmp_pth, 'emi_phosphor_out.out'), 'w') as f2:
            subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'emi_phosphor.inp'], check=True, stdout=f2, stderr=f1)
        with open(Path(tmp_pth, 'emi_phosphor.spectrum'), 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                line = line.strip().split()
                if len(line) == 4:
                    spectrum_x.append(float(line[0])); spectrum_y.append(float(line[1]))
        result['emi_phosphor_wavelength'] = 10**7 / np.array(spectrum_x); result['emi_phosphor_fosc'] = np.array(spectrum_y)
        kr = None 
        with open(Path(tmp_pth, 'emi_phosphor_out.out'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) > 4 and line[0] == 'The' and line[1] == 'calculated' and line[2] == 'phosphorescence':
                    kr = float(line[-2])
        result['kr'] = kr if kr is not None else None
    else:
        return 
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return result

def ic_calculator(args:argparse.Namespace):
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method; sov = args.sov
    freq_file_0 = args.freq_file_0; freq_file_1 = args.freq_file_1
    nproc = args.nproc

    with open(f'{in_pth}/result.pkl', 'rb') as f:
        result = pickle.load(f)
    hessian_s0 = result.get('hessian_S0', None); hessian_s1 = result.get('hessian_S1', None)
    orca_2_gau.hessian_convert(hessian_s0, f'{tmp_pth}/{freq_file_0}', len(result['symbol']), result['s0opt_coord'], result['symbol'], 1)
    orca_2_gau.hessian_convert(hessian_s1, f'{tmp_pth}/{freq_file_1}', len(result['symbol']), result['s1opt_coord'], result['symbol'], 1)
    
    keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF ESD(IC)', '%maxcore 3000', f'%pal nprocs {nproc} end', '%TDDFT', \
                '  TDA FALSE', '  NROOTS 5', 'IROOT 1', 'nacme true', 'etf true', 'END', f'%ESD', f'  GSHESSIAN "{freq_file_0}"', \
                f'  ESHESSIAN "{freq_file_1}"', '  TCUTFREQ 150', '  LINEW 10', '  INLINEW 250', '  usej true', 'END', \
                '* xyz 0 1', '']
     
    with open(Path(tmp_pth, 'ic.inp'), 'w') as f:
        f.write('\n'.join(keywords))
        for ii in range(len(result['symbol'])):
            f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], result['s0opt_coord'][ii][0], result['s0opt_coord'][ii][1], result['s0opt_coord'][ii][2]))
        f.write('end\n')
    # 运行任务 
    with open(Path(tmp_pth, 'ic_err.out'), 'w') as f1, open(Path(tmp_pth, 'ic_out.out'), 'w') as f2:
        subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'ic.inp'], check=True, stdout=f2, stderr=f1)
    kic = None 
    with open(Path(tmp_pth, 'ic_out.out'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 5 and line[0] == 'The' and line[1] == 'calculated' and line[2] == 'internal':
                kic = float(line[-2])
    result['kic'] = kic if kic is not None else None
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return result 

def soc_calculator(args:argparse.Namespace):
    # 计算soc 
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method; sov = args.sov
    nproc = args.nproc 
    
    with open(f'{in_pth}/result.pkl', 'rb') as f:
        result = pickle.load(f)
        coord = np.array(result['t1opt_coord'])
        symbol = result['symbol']
    ss = symbol[0]
    keywords = [f'! {func} d3bj ZORA ZORA-def2-SVP SARC/J RI-SOMF(1X) tightSCF nopop', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                '%TDDFT', '  TDA FALSE', '  NROOTS 25', 'dosoc true', 'end', f'%basis newgto {ss} "SARC-ZORA-SVP" end', 'end', '*xyz 0 1', '']

    with open(Path(tmp_pth, 'soc.inp'), 'w') as f:
        f.write('\n'.join(keywords))
        for ii in range(len(result['symbol'])):
            f.write('%s %.8f %.8f %.8f\n' % (result['symbol'][ii], coord[ii][0], coord[ii][1], coord[ii][2]))
        f.write('end\n')
    # osc计算
    with open(Path(tmp_pth, 'soc_err.out'), 'w') as f1, open(Path(tmp_pth, 'soc_out.out'), 'w') as f2:
        subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'soc.inp'], check=True, stdout=f2, stderr=f1)
    
    # 获取soc 
    with open(Path(tmp_pth, 'soc_out.out'),'r') as fp:
        lines = fp.readlines()
        iline = [i for i in range(len(lines)) if "SOCME" in lines[i].split(" ")]
        assert len(iline) == 1
        iline = iline.pop()+5
        content = lines[iline].split(" ")
        content = [float(ic) for ic in content if ic not in ["", ",", ")", "(", ")\n"]]
        content = content[2:]
        content = [ic**2 for ic in content if ic != 0]
        soc = np.sqrt(sum(content) / 3)
    result['soc_s0t1'] = soc
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return

def isc_calculator(args:argparse.Namespace):
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method; sov = args.sov
    freq_file_0 = args.freq_file_0; freq_file_1 = args.freq_file_1
    nproc = args.nproc
    with open(f'{in_pth}/result.pkl','rb') as f:
        result = pickle.load(f)
    hessian_s0 = result.get('hessian_S0', None); hessian_t1 = result.get('hessian_T1', None)
    orca_2_gau.hessian_convert(hessian_s0, f'{tmp_pth}/{freq_file_0}', len(result['symbol']), result['s0opt_coord'], result['symbol'], 1)
    orca_2_gau.hessian_convert(hessian_t1, f'{tmp_pth}/{freq_file_1}', len(result['symbol']), result['t1opt_coord'], result['symbol'], 3)
    soc = result.get('soc_s0t1', None) / 219474.63
    if 'e_s0_min' in result.keys():
        e_s0_min = result.get('e_s0_min', None)[0]; e_t1_min = result.get('e_t1_min',None)[0]
    else:
        e_s0_min = result.get('e_s0_s0', None)[0]; e_t1_min = result.get('e_t1_td',None)[0]
    
    e_ad = (e_t1_min - e_s0_min) * 219474.6665   
    keywords = [f'%maxcore 3000', f'%pal nprocs 30 end', f'!ESD(ISC) NOITER', f'%ESD', f'  ISCISHESSIAN "{freq_file_1}"', f'  ISCFSHESSIAN "{freq_file_0}"', \
                f'  DELE {e_ad}', f'  SOCME 0.0, {soc}', 'END', '*xyz 0 1']
    coord = result.get('s0opt_coord', None); symbol = result.get('symbol', None)
    with open(f'{tmp_pth}/isc.inp', 'w') as f:
        f.write('\n'.join(keywords)); f.write('\n')
        for ii in range(len(symbol)):
            f.write('%s %.8f %.8f %.8f\n' % (symbol[ii], coord[ii][ 0], coord[ii][1], coord[ii][2]))
        f.write('end\n')
    with open(f'{tmp_pth}/isc_err.out', 'w') as f1, open(f'{tmp_pth}/isc_out.out', 'w') as f2:
        subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'isc.inp'], check=True, stdout=f2, stderr=f1)
    k_isc = None 
    with open(f'{tmp_pth}/isc_out.out', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 5 and line[0] == 'The' and line[1] == 'calculated' and line[2] == 'ISC':
                k_isc = float(line[-2])
    result['k_isc'] = k_isc
    with open(f'{out_pth}/result.pkl', 'wb') as f:
        pickle.dump(result, f)
    return 

def opt_calculator(args:argparse.Namespace):
    '''
    name.xyz 恰好为优化后的结构
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method; sov = args.sov
    nproc = args.nproc 
    if os.path.exists(f'{out_pth}/result.pkl'):
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}; result['Error'] = []  # 错误信息
    os.makedirs(tmp_pth, exist_ok=True)

    if electronic_state == 'S1':
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                    '%tddft', 'nroots 3', 'TDA false', 'iroot 1', 'end', '', '* xyz 0 1']
    elif electronic_state == 'S0':
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                    '', '* xyz 0 1']
    elif electronic_state == 'S0_freq':
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF freq', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                    '', '* xyz 0 1']
    elif electronic_state == 'Cation':
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', '', '* xyz 1 2']
    elif electronic_state == 'T1': # t1计算也可以 0 3 
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                    '%tddft', 'tda false', 'nroots 5', 'iroot 1', 'triplets true', 'irootmult triplet', 'end', '', '* xyz 0 1']
    elif electronic_state == 'T1_uhf': # t1计算也可以 0 3 
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF freq', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                    '', '* xyz 0 3']
    elif electronic_state == 'T2':
        keywords = [f'! opt {func} d3bj def2-svp def2/J RIJCOSX tightSCF', f'%geom', 'MaxIter 80', 'end', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                    '%tddft', 'tda false', 'nroots 5', 'iroot 2', 'triplets true', 'irootmult triplet', 'end', '', '* xyz 0 1']
    
    with open(Path(out_pth, 'result.pkl'), 'rb') as f:
        data = pickle.load(f)
        # 读取坐标和符号
        if electronic_state == 'S0' or electronic_state == 'S0_freq':
            coord = np.array(data['coord_init'])
            symbol = data['symbol']
        elif electronic_state == 'S1' or electronic_state == 'T1' or electronic_state == 'T2' or electronic_state == 'Cation' or electronic_state == 'T1_uhf':
            coord = np.array(data['s0opt_coord'])
            symbol = data['symbol']
        
    result['symbol'] = symbol
    result['converge'] = True  if 'converge' not in result.keys() else result['converge']
    if electronic_state == 'S0' or electronic_state == 'S0_freq':
        f_name = Path(tmp_pth, 's0_opt.inp')
        geometry_2_input.geom_2_inp(f_name, coord, symbol, keywords)
        if 's0opt_coord' in result.keys():
            return
        try:
            with open(Path(tmp_pth, 's0opt_err.out'), 'w') as f1, open(Path(tmp_pth, 's0opt_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's0_opt.inp'], check=True, stdout=f2, stderr=f1)
            # 读取opt的坐标
            s0_opt_coord = [] 
            with open(Path(tmp_pth, 's0_opt.xyz'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split()
                    if len(line) == 4:
                        s0_opt_coord.append([float(x) for x in line[1:]])
            
            with open(f'{tmp_pth}/s0opt_out.out', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 5 and line[3] == 'not' and line[4] == 'converge':
                        result['converge'] = False
            result['s0opt_coord'] = np.array(s0_opt_coord)  # 保存s0态的优化后的坐标
            if electronic_state == 'S0_freq':
                with open(Path(tmp_pth, 's0_opt.hess'), 'r') as f:
                    lines = f.readlines(); read = False; hessian = np.zeros((len(symbol)*3, len(symbol)*3))
                    hessian_idx = 0 
                    for idx, line in enumerate(lines):
                        line = line.strip().split()
                        if len(line) > 0 and line[0] == '$hessian':
                            read = True; hessian_idx = idx 
                        if len(line) > 0 and (line[0] == '$vibrational_frequencies' or line[0] == '$atoms'):
                            read = False
                        if read == True:
                            if len(line) > 1 and float(line[1]).is_integer() == False:
                                col_idx = int(line[0])
                                for u in range(len(line)-1):
                                    hessian[col_idx, row_idx[u]] = float(line[1 + u])
                            elif (len(line) == 1 and idx > hessian_idx + 3 and float(line[0]).is_integer() == True) or (len(line) > 1 and float(line[1]).is_integer() == True):
                                row_idx = [int(x) for x in line[:]]
                result['hessian_S0'] = np.array(hessian)  # 保存S0态的Hessian矩阵
        except:
            traceback.print_exc()
            result['converge'] = False

    elif electronic_state == 'S1':
        try:
            f_name = Path(tmp_pth, 's1_opt.inp')
            geometry_2_input.geom_2_inp(f_name, coord, symbol, keywords)
            if 's1opt_coord' in result.keys():
                return 
            with open(Path(tmp_pth, 's1opt_err.out'), 'w') as f1, open(Path(tmp_pth, 's1opt_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's1_opt.inp'], check=True, stdout=f2, stderr=f1)
            s1_opt_coord = []
            with open(Path(tmp_pth, 's1_opt.xyz'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split()
                    if len(line) == 4:
                        s1_opt_coord.append([float(x) for x in line[1:]])
            with open(f'{tmp_pth}/s1opt_out.out', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 5 and line[3] == 'not' and line[4] == 'converge':
                        result['converge'] = False
            result['s1opt_coord'] = np.array(s1_opt_coord)  # 保存s1态的优化后的坐标
        except:
            traceback.print_exc()
            result['converge'] = False

    elif electronic_state == 'Cation':
        try:
            f_name = Path(tmp_pth, 'cation_opt.inp')
            geometry_2_input.geom_2_inp(f_name, coord, symbol, keywords)
            if 'cationopt_coord' in result.keys():
                return 
            with open(Path(tmp_pth, 'cationopt_err.out'), 'w') as f1, open(Path(tmp_pth, 'cationopt_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'cation_opt.inp'], check=True, stdout=f2, stderr=f1)
            cation_opt_coord = []
            with open(Path(tmp_pth, 'cation_opt.xyz'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split()
                    if len(line) == 4:
                        cation_opt_coord.append([float(x) for x in line[1:]])
            with open(f'{tmp_pth}/cationopt_out.out', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 5 and line[3] == 'not' and line[4] == 'converge':
                        result['converge'] = False
            result['cationopt_coord'] = np.array(cation_opt_coord)  # 保存cation态的优化后的坐标
        except:
            traceback.print_exc()
            result['converge'] = False
    
    elif electronic_state == 'T1' or electronic_state == 'T1_uhf':
        try:
            f_name = Path(tmp_pth, 't1_opt.inp')
            geometry_2_input.geom_2_inp(f_name, coord, symbol, keywords)
            if 't1opt_coord' in result.keys():
                return 
            with open(Path(tmp_pth, 't1opt_err.out'), 'w') as f1, open(Path(tmp_pth, 't1opt_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 't1_opt.inp'], check=True, stdout=f2, stderr=f1)
            t1_opt_coord = []
            with open(Path(tmp_pth, 't1_opt.xyz'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split()
                    if len(line) == 4:
                        t1_opt_coord.append([float(x) for x in line[1:]])
            with open(f'{tmp_pth}/t1opt_out.out', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 5 and line[3] == 'not' and line[4] == 'converge':
                        result['converge'] = False
            result['t1opt_coord'] = np.array(t1_opt_coord)  # 保存T1态的优化后的坐标

            if electronic_state == 'T1_uhf':
                with open(Path(tmp_pth, 't1_opt.hess'), 'r') as f:
                    lines = f.readlines(); read = False; hessian = np.zeros((len(symbol)*3, len(symbol)*3))
                    hessian_idx = 0
                    for idx, line in enumerate(lines):
                        line = line.strip().split()
                        if len(line) > 0 and line[0] == '$hessian':
                            read = True; hessian_idx = idx
                        if len(line) > 0 and (line[0] == '$vibrational_frequencies' or line[0] == '$atoms'):
                            read = False
                        if read == True:
                            if len(line) > 1 and float(line[1]).is_integer() == False:
                                col_idx = int(line[0])
                                for u in range(len(line)-1):
                                    hessian[col_idx, row_idx[u]] = float(line[1 + u])
                            elif (len(line) == 1 and idx > hessian_idx +3 and float(line[0]).is_integer() == True) or (len(line) > 1 and float(line[1]).is_integer() == True):
                                row_idx = [int(x) for x in line[:]]
                result['hessian_T1'] = np.array(hessian)  # 保存T1_uhf态的Hessian矩阵
        except:
            traceback.print_exc()
            result['converge'] = False
        
    elif electronic_state == 'T2':
        try:
            f_name = Path(tmp_pth, 't2_opt.inp')
            geometry_2_input.geom_2_inp(f_name, coord, symbol, keywords)
            if 't2opt_coord' in result.keys():
                return 
            with open(Path(tmp_pth, 't2opt_err.out'), 'w') as f1, open(Path(tmp_pth, 't2opt_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 't2_opt.inp'], check=True, stdout=f2, stderr=f1)
            t2_opt_coord = []
            with open(Path(tmp_pth, 't2_opt.xyz'), 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split()
                    if len(line) == 4:
                        t2_opt_coord.append([float(x) for x in line[1:]])
            with open(f'{tmp_pth}/t2opt_out.out', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 5 and line[3] == 'not' and line[4] == 'converge':
                        result['converge'] = False
            result['t2opt_coord'] = np.array(t2_opt_coord)  # 保存T2态的优化后的坐标
        except:
            traceback.print_exc()
            result['converge'] = False

    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    
    #with open(Path(out_pth, 'result.csv'), 'w') as f:
    #    df = pd.DataFrame(result)
    #    df.to_csv(f, index=False)
    return 


def orca_single_calculator(args:argparse.Namespace):
    '''
    orca软件的单点性质计算，暂时未用到
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method; sov = args.sov
    nproc = args.nproc
    if os.path.exists(f'{out_pth}/result.pkl'):
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}; result['Error'] = []  # 错误信息
    os.makedirs(tmp_pth, exist_ok=True)
    try:
        if electronic_state == 'S0':
            keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', '', '* xyz 0 1']
            coord = result['s0opt_coord']; symbol = result['symbol']
            if 'homo' in result.keys():
                return 
            geometry_2_input.geom_2_inp(Path(tmp_pth, 's0_ground.inp'), coord, symbol, keywords)
        elif electronic_state == 'S0_td':
            keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', '%tddft', 'nroots 20',
                        'TDA false', 'end', '', '* xyz 0 1'] 
            coord = result['s0opt_coord']; symbol = result['symbol']
            if 'abs_wavelength' in result.keys():
                return 
            geometry_2_input.geom_2_inp(Path(tmp_pth, 's0_td.inp'), coord, symbol, keywords)
        elif electronic_state == 'S1_td':
            keywords= [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', '%tddft', 'nroots 10',
                        'TDA false', 'end', '', '* xyz 0 1'] 
            coord = result['s1opt_coord']; symbol = result['symbol']
            if 'emi_wavelength' in result.keys():
                return
            geometry_2_input.geom_2_inp(Path(tmp_pth, 's1_td.inp'), coord, symbol, keywords)

        elif electronic_state == 'Cation':
            keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', '', '* xyz 1 2']
            coord = result['cationopt_coord']; symbol = result['symbol']
            if 'cation_energy' in result.keys():
                return
            geometry_2_input.geom_2_inp(Path(tmp_pth, 'cation.inp'), coord, symbol, keywords)
        
        elif electronic_state == 'T1':
            keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                        '%tddft', 'tda false', 'nroots 5', 'iroot 1', 'triplets true', 'irootmult triplet', 'end', '', '* xyz 0 1']
            coord = result['t1opt_coord']; symbol = result['symbol']
            if 'e_t1_min' in result.keys():
                return
            geometry_2_input.geom_2_inp(Path(tmp_pth, 't1.inp'), coord, symbol, keywords)
        
        elif electronic_state == 'T1_uhf':
            keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', '', '* xyz 0 3']
            coord = result['t1opt_coord']; symbol = result['symbol']
            if 'e_t1_min' in result.keys():
                return
            geometry_2_input.geom_2_inp(Path(tmp_pth, 't1_uhf.inp'), coord, symbol, keywords)
        
        
        elif electronic_state == 'T2':
            keywords = [f'! {func} d3bj def2-svp def2/J RIJCOSX tightSCF', '%maxcore 3000', f'%pal nprocs {nproc} end', \
                        '%tddft', 'tda false', 'nroots 5', 'iroot 2', 'triplets true', 'irootmult triplet', 'end', '', '* xyz 0 1']
            coord = result['t2opt_coord']; symbol = result['symbol']
            if 'e_t2_min' in result.keys():
                return
            geometry_2_input.geom_2_inp(Path(tmp_pth, 't2.inp'), coord, symbol, keywords)
        
        if electronic_state == 'S0':
            with open(Path(tmp_pth, 's0_ground_err.out'), 'w') as f1, open(Path(tmp_pth, 's0_ground_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's0_ground.inp'], check=True, stdout=f2, stderr=f1)
            with open(Path(tmp_pth, 's0_ground_out.out'), 'r') as f:
                lines = f.readlines(); read_data = False; homo = []; lumo = []; e_s0_min = None 
                for idx, line in enumerate(lines):
                    line = line.strip().split()
                    if len(line) > 3 and line[0] == '*Only' and line[1] == 'the' and line[2] == 'first':
                        read_data = False
                    if read_data:
                        if float(line[1]) > 1.:
                            homo.append(float(line[3]))
                        if float(line[1]) < 1.:
                            lumo.append(float(line[3]))
                    if len(line) == 4 and line[0] == 'NO' and line[1] == 'OCC' and line[2] == 'E(Eh)':
                        read_data = True
                    if len(line) == 5 and line[0] == 'FINAL' and line[1] == 'SINGLE' and line[2] == 'POINT':
                        e_s0_min = float(line[-1])
            result['e_s0_min'] = e_s0_min 
            if len(homo) > 0:
                result['homo'] = homo[-1]; result['lumo'] = lumo[0]

        if electronic_state == 'S0_td':
            with open(Path(tmp_pth, 's0_td_err.out'),'w') as f1, open(Path(tmp_pth,'s0_td_out.out'),'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's0_td.inp'], check=True, stdout=f2, stderr=f1)
            wavelength = []; fosc= []
            with open(Path(tmp_pth, 's0_td_out.out'), 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    line = line.strip().split()
                    if len(line) > 4 and line[0] == '0-1A' and line[1] == '->' and line[2] == '1-1A' and len(wavelength) < 1:
                        wavelength.append(float(line[5])); fosc.append(float(line[6]))
                        wavelength.append(float(lines[idx+1].strip().split()[5])); fosc.append(float(lines[idx+1].strip().split()[6]))
                        wavelength.append(float(lines[idx+2].strip().split()[5])); fosc.append(float(lines[idx+2].strip().split()[6]))
                        wavelength.append(float(lines[idx+3].strip().split()[5])); fosc.append(float(lines[idx+3].strip().split()[6]))
                        wavelength.append(float(lines[idx+4].strip().split()[5])); fosc.append(float(lines[idx+4].strip().split()[6]))
                        wavelength.append(float(lines[idx+5].strip().split()[5])); fosc.append(float(lines[idx+5].strip().split()[6]))
                        wavelength.append(float(lines[idx+6].strip().split()[5])); fosc.append(float(lines[idx+6].strip().split()[6]))
                        wavelength.append(float(lines[idx+7].strip().split()[5])); fosc.append(float(lines[idx+7].strip().split()[6]))
                        wavelength.append(float(lines[idx+8].strip().split()[5])); fosc.append(float(lines[idx+8].strip().split()[6]))
                        wavelength.append(float(lines[idx+9].strip().split()[5])); fosc.append(float(lines[idx+9].strip().split()[6]))
                        wavelength.append(float(lines[idx+10].strip().split()[5])); fosc.append(float(lines[idx+10].strip().split()[6]))

            if wavelength is not None and fosc is not None:
                result['abs_wavelength'] = wavelength; result['abs_fosc'] = fosc

        if electronic_state == 'S1_td':
            with open(Path(tmp_pth, 's1_td_err.out'),'w') as f1, open(Path(tmp_pth,'s1_td_out.out'),'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's1_td.inp'], check=True, stdout=f2, stderr=f1)
            wavelength = None; fosc= None; e_s1_min = None
            with open(Path(tmp_pth, 's1_td_out.out'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 4 and line[0] == '0-1A' and line[1] == '->' and line[2] == '1-1A' and wavelength is None:
                        wavelength = float(line[5]); fosc = float(line[6])
                    if len(line) == 5 and line[0] == 'FINAL' and line[1] == 'SINGLE' and line[2] == 'POINT':
                        e_s1_min = float(line[-1])
            result['e_s1_min'] = e_s1_min  # 保存s1态的能量
            if wavelength is not None and fosc is not None:
                result['emi_wavelength'] = wavelength; result['emi_fosc'] = fosc

        if electronic_state == 'Cation':
            # FINAL SINGLE POINT ENERGY 
            with open(Path(tmp_pth, 'cation_err.out'), 'w') as f1, open(Path(tmp_pth, 'cation_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 'cation.inp'], check=True, stdout=f2, stderr=f1)
            with open(Path(tmp_pth, 'cation_out.out'), 'r') as f:
                lines = f.readlines(); e_cation_min = None
                # 读取cation 能量    
                for line in lines:
                    line = line.strip().split()
                    if len(line) == 5 and line[0] == 'FINAL' and line[1] == 'SINGLE' and line[2] == 'POINT':
                        e_cation_min = float(line[-1])
                result['cation_energy'] = e_cation_min  # 保存cation态的能量


        if electronic_state == 'T1':
            with open(Path(tmp_pth, 't1_err.out'), 'w') as f1, open(Path(tmp_pth, 't1_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 't1.inp'], check=True, stdout=f2, stderr=f1)
            with open(Path(tmp_pth, 't1_out.out'), 'r') as f:
                lines = f.readlines(); e_t1_min = None
                # 读取T1 能量
                for line in lines:
                    line = line.strip().split()
                    if len(line) == 5 and line[0] == 'FINAL' and line[1] == 'SINGLE' and line[2] == 'POINT':
                        e_t1_min = float(line[-1])
                result['e_t1_min'] = e_t1_min  # 保存T1态的能量

        if electronic_state == 'T1_uhf':
            with open(Path(tmp_pth, 't1_uhf_err.out'), 'w') as f1, open(Path(tmp_pth, 't1_uhf_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 't1_uhf.inp'], check=True, stdout=f2, stderr=f1)
            with open(Path(tmp_pth, 't1_uhf_out.out'), 'r') as f:
                lines = f.readlines(); e_t1_min = None
                # 读取T1 能量
                for line in lines:
                    line = line.strip().split()
                    if len(line) == 5 and line[0] == 'FINAL' and line[1] == 'SINGLE' and line[2] == 'POINT':
                        e_t1_min = float(line[-1])
                result['e_t1_min'] = e_t1_min


        if electronic_state == 'T2':
            with open(Path(tmp_pth, 't2_err.out'), 'w') as f1, open(Path(tmp_pth, 't2_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 't2.inp'], check=True, stdout=f2, stderr=f1)
            with open(Path(tmp_pth, 't2_out.out'), 'r') as f:
                lines = f.readlines(); e_t2_min = None
                # 读取T2 能量
                for line in lines:
                    line = line.strip().split()
                    if len(line) == 5 and line[0] == 'FINAL' and line[1] == 'SINGLE' and line[2] == 'POINT':
                        e_t2_min = float(line[-1])
                result['e_t2_min'] = e_t2_min  # 保存T2态的能量
                
    except:
        print(traceback.format_exc())
        result['converge'] = False  
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
    return 

def gen_electronic_e_slurm(args:argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); orca_env = args.orca_env; platform_env = args.platform_env
    script_pth = Path(args.script_pth); out_pth = Path(args.out_pth)
    nproc = args.nproc
    calc_files = glob(f'{in_pth}/*/result.pkl')
    for f_name in calc_files:
        f_dir = os.path.dirname(f_name)
        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(orca_env.strip().split(';')) if orca_env else []
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}', f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S0 --sov CPCM(CH2Cl2)',
        f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S1 --sov CPCM(CH2Cl2)',
        f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state T1 --sov CPCM(CH2Cl2)',
        f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state T2 --sov CPCM(CH2Cl2)',
        f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state Cation --sov CPCM(CH2Cl2)',
        f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S0 --sov CPCM(CH2Cl2)',
        f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S0_td --sov CPCM(CH2Cl2)',
        f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S1_td --sov CPCM(CH2Cl2)', 
        f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state Cation --sov CPCM(CH2Cl2)',
        f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state T1 --sov CPCM(CH2Cl2)',
        f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state T2 --sov CPCM(CH2Cl2)',
        ])
        slurm_txt.extend([f'cp {tmp_pth}/mol_{idx}/*out.out {f_dir}'])
        slurm_txt.extend([f'cp {tmp_pth}/mol_{idx}/*.xyz {f_dir}'])  # 保存结果文件
        slurm_txt.extend([f'rm -rf {tmp_pth}/mol_{idx}']) # 清理临时文件夹    
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def gen_property_slurm(args:argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); orca_env = args.orca_env; platform_env = args.platform_env
    script_pth = Path(args.script_pth); out_pth = Path(args.out_pth)
    nproc = args.nproc
    calc_files = glob(f'{in_pth}/*/result.pkl'); samp_idx = np.arange(0, len(calc_files), 10)
    calc_files = [calc_files[i] for i in samp_idx]  # 每10个采样一个
    
    for f_name in calc_files:
        #print(f_name)
        f_dir = os.path.dirname(f_name)
        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(orca_env.strip().split(';')) if orca_env else []
        # 计算工作流调用
        method = None 
        # 读取参数确定method 
        
        with open(f'{in_pth}/frame_{idx}/result.pkl', 'rb') as f:
            data = pickle.load(f)
            if 'name' in data.keys():
                name = data['name'].split('_')[0]
                if name == 'A' or name == 'B' or name == 'C' or name == 'D' or name == 'E':
                    method = 'REVPBE38'
                elif name == 'F' or name == 'H':
                    method = 'PBE0'
                elif name == 'G' or name == 'I' or name == 'J' or name == 'K' or name == 'L' or name == 'M':
                    method = 'B3LYP'
                else:
                    method = 'PBE0'
            else:
                method = 'PBE0'
            
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}',f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S0', 
                      f'python {script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx}  --method "PBE0" --electronic_state S1',
                      f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "PBE0" --electronic_state S0',
                      f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "{method}" --electronic_state S0_td',
                      f'python {script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "{method}" --electronic_state S1_td'])   
        slurm_txt.extend([f'cp {tmp_pth}/mol_{idx}/*out.out {f_dir}'])
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
    parser_gen_input.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for xyz files')

    
    # subparser for structure optimization
    parser_structure_opt = subparsers.add_parser('structure_opt', help='Run structure optimization')
    parser_structure_opt.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_structure_opt.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')
    parser_structure_opt.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_structure_opt.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_structure_opt.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')
    parser_structure_opt.add_argument('--sov', type=str, default='water', help='Solvent for optimization, default is water')
    parser_structure_opt.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for orca single calculator
    parser_orca_single_calculator = subparsers.add_parser('orca_single_calculator', help='Run single point calculation with ORCA')
    parser_orca_single_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_orca_single_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')
    parser_orca_single_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_orca_single_calculator.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_orca_single_calculator.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')
    parser_orca_single_calculator.add_argument('--sov', type=str, default='water', help='Solvent for optimization, default is water')
    parser_orca_single_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for spectrum calculator 
    parser_spectrum_calculator = subparsers.add_parser('spectrum_calculator', help='Run spectrum calculation')
    parser_spectrum_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_spectrum_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for optimization results')
    parser_spectrum_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for optimization')
    parser_spectrum_calculator.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')
    parser_spectrum_calculator.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_spectrum_calculator.add_argument('--sov', type=str, default='water', help='Solvent for optimization, default is water')
    parser_spectrum_calculator.add_argument('--freq_file_0', type=str, default='/public/home/chengz/FunMG/freq_0.out', help='Frequency file for S0 state')
    parser_spectrum_calculator.add_argument('--freq_file_1', type=str, default='/public/home/chengz/FunMG/freq_1.out', help='Frequency file for S1 state')
    parser_spectrum_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for ic calculator
    parser_ic_calculator = subparsers.add_parser('ic_calculator', help='Run internal conversion calculation')
    parser_ic_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_ic_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for optimization results')
    parser_ic_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for optimization')
    parser_ic_calculator.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')
    parser_ic_calculator.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_ic_calculator.add_argument('--sov', type=str, default='water', help='Solvent for optimization, default is water')
    parser_ic_calculator.add_argument('--freq_file_0', type=str, default='/public/home/chengz/FunMG/freq_0.out', help='Frequency file for S0 state')
    parser_ic_calculator.add_argument('--freq_file_1', type=str, default='/public/home/chengz/FunMG/freq_1.out', help='Frequency file for S1 state')
    parser_ic_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for isc calculator 
    parser_isc_calculator = subparsers.add_parser('isc_calculator', help='Run inter-system crossing calculation')
    parser_isc_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_isc_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for optimization results')
    parser_isc_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for optimization')
    parser_isc_calculator.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')
    parser_isc_calculator.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_isc_calculator.add_argument('--sov', type=str, default='water', help='Solvent for optimization, default is water')
    parser_isc_calculator.add_argument('--freq_file_0', type=str, default='/public/home/chengz/FunMG/freq_0.out', help='Frequency file for S0 state')
    parser_isc_calculator.add_argument('--freq_file_1', type=str, default='/public/home/chengz/FunMG/freq_1.out', help='Frequency file for S1 state')
    parser_isc_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for soc calculator
    parser_soc_calculator = subparsers.add_parser('soc_calculator', help='Run spin-orbit coupling calculation')
    parser_soc_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_soc_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for optimization results')
    parser_soc_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for optimization')
    parser_soc_calculator.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')
    parser_soc_calculator.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_soc_calculator.add_argument('--sov', type=str, default='water', help='Solvent for optimization, default is water')
    parser_soc_calculator.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for generating property slurm
    parser_gen_property_slurm = subparsers.add_parser('gen_property_slurm', help='Generate property calculation slurm script')
    parser_gen_property_slurm.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_gen_property_slurm.add_argument('--slurm_task_pth', type=str, default='/public/home/chengz/FunMG/task', help='Input path for slurm')
    parser_gen_property_slurm.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for software calculation')
    parser_gen_property_slurm.add_argument('--orca_env', type=str,  default='export GAUSS_SCRDIR=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Environment for ORCA')
    parser_gen_property_slurm.add_argument('--platform_env', type=str, default='#SBATCH -p kshcnormal', help='Environment for Sugon')
    parser_gen_property_slurm.add_argument('--script_pth', type=str, default='/public/home/chengz/FunMG/job_adv.py', help='Path for job_adv.py')
    parser_gen_property_slurm.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for optimization results')
    parser_gen_property_slurm.add_argument('--nproc', type=int, default=8, help='Number of processors to use')

    # subparser for generating electronic energy slurm
    parser_gen_property_slurm = subparsers.add_parser('gen_electronic_slurm', help='Generate property calculation slurm script')
    parser_gen_property_slurm.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG', help='Input path for com')
    parser_gen_property_slurm.add_argument('--slurm_task_pth', type=str, default='/public/home/chengz/FunMG/task', help='Input path for slurm')
    parser_gen_property_slurm.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Temporary path for software calculation')
    parser_gen_property_slurm.add_argument('--orca_env', type=str,  default='export GAUSS_SCRDIR=/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID', help='Environment for ORCA')
    parser_gen_property_slurm.add_argument('--platform_env', type=str, default='#SBATCH -p kshcnormal', help='Environment for Sugon')
    parser_gen_property_slurm.add_argument('--script_pth', type=str, default='/public/home/chengz/FunMG/job_adv.py', help='Path for job_adv.py')
    parser_gen_property_slurm.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG', help='Output path for optimization results')
    parser_gen_property_slurm.add_argument('--nproc', type=int, default=8, help='Number of processors to use')
    
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
    try:
        if args.command == 'gen_input':
            # 全部为生成slurm 
            gen_data(args)
        
        if args.command == 'structure_opt':
            # 结构优化计算
            opt_calculator(args)
        
        if args.command == 'orca_single_calculator':
            # 单点性质计算
            orca_single_calculator(args)

        if args.command == 'spectrum_calculator':
            # 光谱计算
            spec_calculator(args)

        if args.command == 'soc_calculator':
            # soc计算
            soc_calculator(args)

        if args.command == 'ic_calculator':
            # 内转换计算
            ic_calculator(args)

        if args.command == 'isc_calculator':
            isc_calculator(args)

        ##################################################
        # 根据需求组装单个任务,并生成相应的slurm 脚本
        ##################################################
        if args.command == 'gen_property_slurm':
            # 生成从分子结构生成-光物理性质计算脚本; 提供软件环境slurm环境；提供计算参数
            os.makedirs(args.slurm_task_pth, exist_ok=True)  # 确保任务路径存在
            gen_property_slurm(args)

        if args.command == 'gen_electronic_slurm':
            # 生成从分子结构生成-光物理性质计算脚本; 提供软件环境slurm环境；提供计算参数
            os.makedirs(args.slurm_task_pth, exist_ok=True)
            gen_electronic_e_slurm(args)

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
    except Exception as e:
        print(traceback.format_exc())

        

if __name__ == '__main__':
    main()

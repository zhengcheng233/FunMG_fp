#!/usr/bin/env python 
from typing import List, Dict, Any
import subprocess
import logging
import fchic 
from ase.data import chemical_symbols, atomic_numbers
import numpy as np 
import json 
'''
包含main函数，解析参数；make_fp函数：批处理初始文件，生成s0的input.com
run_fp函数：根据任务不同运行相应的任务：包括s0, s1, t1 的结构优化，单点计算，光谱生成
post_fp函数：根据任务不同，处理输出文件，生成最终结果，可能是json文件，可能是光谱数据
'''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gaussian(x, mu, fwhm, amplitude):
    sigma = fwhm /(2 * np.sqrt(2 * np.log(2)))  # 展宽系数
    return 28700 * amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def gencom(*args, **kwargs):
    with open(f'{args[0]}.com','w') as f:
        f.write(f'%chk={args[0]}.chk\n')
        f.write(f'%mem={args[8]}\n')
        f.write(f'%nproc={args[7]}\n')
        f.write(f'# {args[5]} {args[6]}\n')
        f.write('\n')
        f.write(f'{args[0]}\n')
        f.write('\n')
        f.write(f'{args[3]} {args[4]}\n')
        for i in range(len(args[1])):
            f.write('%s %.6f %.6f %.6f \n' % (args[1][i], args[2][i][0], args[2][i][1], args[2][i][2]))
        f.write('\n')
    return 

def make_fp(jparm:Dict, mparm:Dict)->Any:
    # 补充缺省值
    input_name = jparm.get('input_name','input.xyz')
    method = jparm.get('method','b3lyp EmpiricalDispersion=GD3BJ')
    basis = jparm.get('basis','def2svp')
    charge = jparm.get('charge','0')
    multiplicity = jparm.get('multiplicity','1')
    nproc = mparm.get('nproc','32')
    mem = mparm.get('mem','64GB')
    # 读取input文件, 目前只支持xyz文件
    coord = []; symbol = []
    with open(input_name, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            l = line.strip().split()
            if len(l) == 4:
                symbol.append(l[0])
                coord.append([float(l[1]), float(l[2]), float(l[3])])
    # 生成s0.com文件
    method += ' opt'
    # 所有生成均按此格式
    gencom('s0', symbol, coord, charge, multiplicity, method, basis, nproc, mem)
    return 

def run_fp(jparm:Dict, mparm:Dict)->Any:
    input_name = jparm.get('input_name','input.xyz')
    method = jparm.get('method','b3lyp EmpiricalDispersion=GD3BJ')
    basis = jparm.get('basis','def2svp')
    charge = jparm.get('charge','0')
    multiplicity = jparm.get('multiplicity','1')
    nproc = mparm.get('nproc','32')
    mem = mparm.get('mem','64GB')
    task_name = jparm.get('task','s0')
    # 运行可能的任务
    if task_name == 's0':
        try:
            result = subprocess.run(['g16', 's0.com'],capture_output=True, text=True)
            result = subprocess.run(['formchk', 's0.chk', 's0.fchk'], capture_output=True, text=True)
        except Exception as e:
            logging.error(f'Error running s0: {e}')

    elif task_name == 's1':
        try:
            # 运行s0的结果文件
            result = subprocess.run(['g16', 's0.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's0.chk', 's0.fchk'], capture_output=True, text=True)
            # 将s0的结果转换为s1的输入文件, 注意采用fchk信息时候，单位是au
            with open('s0.fchk', 'r') as f:
                coord_s0_opt = fchic.deck_load(f, "Current cartesian coordinates")
                species_s0_opt = fchic.deck_load(f, "Atomic numbers")
                symbol_s0_opt = [chemical_symbols[i] for i in species_s0_opt]
                coord_s0_opt = np.array(coord_s0_opt)
                coord_s0_opt = coord_s0_opt.reshape((-1,3)) * 0.5291772
            # 生成s0下的td单点计算
            method += ' td=(nstates=30)'
            gencom('s0_td_single', symbol_s0_opt, coord_s0_opt, charge, multiplicity, method, basis, nproc, mem)
            result = subprocess.run(['g16', 's0_td_single.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's0_td_single.chk'], capture_output=True, text=True)
            # 生成s1的输入文件
            method += ' opt'
            gencom('s1', symbol_s0_opt, coord_s0_opt, charge, multiplicity, method, basis, nproc, mem)
            # 运行s1的任务
            result = subprocess.run(['g16', 's1.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's1.chk'], capture_output=True, text=True)
            # 将s1的结构文件提取出来，计算基态下的单点能量
            with open('s1.fchk', 'r') as f:
                coord_s1_opt = fchic.deck_load(f, "Current cartesian coordinates")
                species_s1_opt = fchic.deck_load(f, "Atomic numbers")
                symbol_s1_opt = [chemical_symbols[i] for i in species_s1_opt]
                coord_s1_opt = np.array(coord_s1_opt)
                coord_s1_opt = coord_s1_opt.reshape((-1,3)) * 0.5291772
            # 生成s1的单点计算
            method = jparm.get('method','b3lyp EmpiricalDispersion=GD3BJ')
            gencom('s1_ground_single', symbol_s1_opt, coord_s1_opt, charge, multiplicity, method, basis, nproc, mem)
            result = subprocess.run(['g16', 's1_ground_single.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's1_ground_single.chk'], capture_output=True, text=True)
        except Exception as e:
            logging.error(f'Error running s1: {e}')

    elif task_name == 't1':
        try:
            # 运行s0的结果文件
            result = subprocess.run(['g16', 's0.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's0.chk', 's0.fchk'], capture_output=True, text=True)
            # 将s0的结果转换为t1的输入文件, 注意采用fchk信息时候，单位是au
            with open('s0.fchk', 'r') as f:
                coord_s0_opt = fchic.deck_load(f, "Current cartesian coordinates")
                species_s0_opt = fchic.deck_load(f, "Atomic numbers")
                symbol_s0_opt = [chemical_symbols[i] for i in species_s0_opt]
                coord_s0_opt = np.array(coord_s0_opt)
                coord_s0_opt = coord_s0_opt.reshape((-1,3)) * 0.5291772
            multiplicity = 3 
            gencom('t1_td_singlet', symbol_s0_opt, coord_s0_opt, charge, multiplicity, method, basis, nproc, mem)
            result = subprocess.run(['g16', 't1_td_singlet.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 't1_td_singlet.chk', 't1_td_singlet.fchk'], capture_output=True, text=True)
            # 生成t1的输入文件
            method += ' opt'
            multiplicity = 3
            gencom('t1', symbol_s0_opt, coord_s0_opt, charge, multiplicity, method, basis, nproc, mem)
            # 运行t1的任务
            result = subprocess.run(['g16', 't1.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 't1.chk', 't1.fchk'], capture_output=True, text=True)
            # 将t1的结构文件提取出来，计算基态下的单点能量
            with open('t1.fchk', 'r') as f:
                coord_t1_opt = fchic.deck_load(f, "Current cartesian coordinates")
                species_t1_opt = fchic.deck_load(f, "Atomic numbers")
                symbol_t1_opt = [chemical_symbols[i] for i in species_t1_opt]
                coord_t1_opt = np.array(coord_t1_opt)
                coord_t1_opt = coord_t1_opt.reshape((-1,3)) * 0.5291772
            # 生成t1的单点计算
            method = jparm.get('method','b3lyp EmpiricalDispersion=GD3BJ')
            multiplicity = 1 
            gencom('t1_ground_single', symbol_t1_opt, coord_t1_opt, charge, multiplicity, method, basis, nproc, mem)
            result = subprocess.run(['g16', 't1_ground_single.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 't1_ground_single.chk', 't1_ground_single.fchk'], capture_output=True, text=True)
        except Exception as e:
            logging.error(f'Error running t1: {e}')

    elif task_name == 'absorption_elec_spec':
        try:
            # 运行s0的结果文件
            result = subprocess.run(['g16', 's0.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's0.chk', 's0.fchk'], capture_output=True, text=True)
            with open('s0.fchk', 'r') as f:
                coord_s0_opt = fchic.deck_load(f, "Current cartesian coordinates")
                species_s0_opt = fchic.deck_load(f, "Atomic numbers")
                symbol_s0_opt = [chemical_symbols[i] for i in species_s0_opt]
                coord_s0_opt = np.array(coord_s0_opt)
                coord_s0_opt = coord_s0_opt.reshape((-1,3)) * 0.5291772
            # 生成s1态的单点计算
            method += ' td=(nstates=30)'
            gencom('s1_single', symbol_s0_opt, coord_s0_opt, charge, multiplicity, method, basis, nproc, mem)
            # 运行s1的单点计算
            result = subprocess.run(['g16', 's1_single.com'], capture_output=True, text=True)
        except Exception as e:
            logging.error(f'Error running absorption_elec_spec: {e}')

    elif task_name == 'emission_ele_spec':
        try:
            # 等同于计算s1 
            result = subprocess.run(['g16', 's0.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's0.chk', 's0.fchk'], capture_output=True, text=True)
            with open('s0.fchk', 'r') as f:
                coord_s0_opt = fchic.deck_load(f, "Current cartesian coordinates")
                species_s0_opt = fchic.deck_load(f, "Atomic numbers")
                symbol_s0_opt = [chemical_symbols[i] for i in species_s0_opt]
                coord_s0_opt = np.array(coord_s0_opt)
                coord_s0_opt = coord_s0_opt.reshape((-1,3)) * 0.5291772
            # 生成s1态的单点计算
            method += ' td=(nstates=10) opt'
            gencom('s1', symbol_s0_opt, coord_s0_opt, charge, multiplicity, method, basis, nproc, mem)
            # 运行s1的任务
            result = subprocess.run(['g16', 's1.com'], capture_output=True, text=True)
            result = subprocess.run(['formchk', 's1.chk'], capture_output=True, text=True)

        except Exception as e:
            logging.error(f'Error running emission_ele_spec: {e}')

    else:
        logging.error(f'Unsupported task: {task_name}')
    return 

def post_fp(jparm:Dict, mparm:Dict)->Any:
    '''
    后处理运行任务, 生成json文件或者光谱数据 
    '''
    input_name = jparm.get('input_name','input.xyz')
    method = jparm.get('method','b3lyp EmpiricalDispersion=GD3BJ')
    basis = jparm.get('basis','def2svp')
    charge = jparm.get('charge','0')
    multiplicity = jparm.get('multiplicity','1')
    nproc = mparm.get('nproc','32')
    mem = mparm.get('mem','64GB')
    task_name = jparm.get('task','s0')

    data = {}
    if task_name == 's0':
        try:
            # 后处理s0, 如果是s0, 只有基态能量相对有用
            with open('s0.fchk', 'r') as f:
                # 由于我们只采用dft方法，所以s0能量就是scf能量
                energy_s0 = fchic.deck_load(f, "SCF Energy")
            data['e_s0'] = energy_s0
            converge = True
            with open('s0.log','r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    converge = True
                else:
                    converge = False
            data['converge'] = converge
            with open('data.json','w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post s0: {e}')
    
    elif task_name == 's1':
        try:
            converge = True 
            # 后处理s0 s1, 重要信息包括发射能，重组能 
            with open('s1.fchk','r') as f:
                # 读取优化后第一激发态能量
                td_s1 = np.array(fchic.deck_load(f, "CIS Energy"))
            with open('s1.log','r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('s1_ground_single.fchk', 'r') as f:
                # 读取优化后结构的基态能量
                ground_s1 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open('s1_ground_single.log','r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('s0.fchk', 'r') as f:
                # 读取基态能量
                ground_s0 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open('s0.log', 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('s0_td_single.fchk', 'r') as f:
                # 读取s0的td能量
                td_s0 = np.array(fchic.deck_load(f, "CIS Energy"))
            with open('s0_td_single.log', 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            # 获取发射能&重组能
            emission_energy = td_s1 - ground_s1
            reorg_energy = np.abs((td_s0 - td_s1)) + np.abs((ground_s1 - ground_s0))
            data['emission'] = list(emission_energy * 27.2114)
            data['reorg'] = list(reorg_energy * 27.2114)
            data['converge'] = converge 
            with open('data.json','w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post s1: {e}')
    
    elif task_name == 't1':
        try:
            converge = True
            # 后处理t1, 重要信息包括发射能，重组能
            with open('t1.fchk', 'r') as f:
                td_t1 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open('t1.log', 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('t1_ground_single.fchk', 'r') as f:
                ground_t1 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open('t1_ground_single.log', 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('s0.fchk', 'r') as f:
                ground_s0 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open('s0.log', 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('t1_td_singlet.fchk', 'r') as f:
                td_s0 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open('t1_td_singlet.log', 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            emission_energy = td_t1 - ground_t1 
            reorg_energy = np.abs((td_s0 - td_t1)) + np.abs((ground_t1 - ground_s0))
            data['emission'] = list(emission_energy * 27.2114)
            data['reorg'] = list(reorg_energy * 27.2114)
            data['converge'] = converge
            with open('data.json','w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post t1: {e}')

    elif task_name == 'absorption_elec_spec':
        try:
            converge = True
            # 后处理吸收光谱，读取吸收能&edme, 然后生成光谱数据    
            e_abs = []; fs = []
            with open('s1_single.log','r') as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 4:
                        if line[0] == 'Excited' and line[1] == 'State':
                            e_abs.append(float(line[4]))
                            fs.append(float(line[-2].split('=')[-1]))
            with open('s0.log','r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open('s1_single.log','r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
                
            fs = np.array(fs); e_abs = np.array(e_abs)
            # 生成光谱数据
            min_energy = 1.24; max_energy = 5.64
            energy_range = np.linspace(min_energy, max_energy, 1000)
            fwhm = 0.5 
            spectrum = np.zeros_like(energy_range)
            for energy, strength in zip(e_abs, fs):
                spectrum += gaussian(energy_range, energy, fwhm, strength)
            # 单位为eV
            data['spectrum_y'] = list(spectrum)
            data['spectrum_x'] = list(energy_range) 
            data['converge'] = converge
            with open('data.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post absorption_elec_spec: {e}')
    
    elif task_name == 'emission_ele_spec':
        try:
            # 后处理发射光谱，读取发射能, 直接采用高斯展宽 
            # 暂时不支持
            print('Not support emission_ele_spec')
        except Exception as e:
            logging.error(f'Error post emission_ele_spec: {e}')
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
    make_fp(jparm, mparm)
    # 运行任务
    run_fp(jparm, mparm)
    # 后处理, 包括生成json文件或者光��数据
    post_fp(jparm, mparm)
    return 

if __name__ == '__main__':
    main()
    

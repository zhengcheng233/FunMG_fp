#!/usr/bin/env python 
from typing import List, Dict, Any
import subprocess
from loguru import logger as logging
import fchic 
from ase.data import chemical_symbols, atomic_numbers
import numpy as np 
import json
import copy
from pathlib import Path
from typings import JobType,G16Input,Atom,Structure,Params

'''
包含main函数，解析参数；make_fp函数：批处理初始文件，生成s0的input.com
run_fp函数：根据任务不同运行相应的任务：包括s0, s1, t1 的结构优化，单点计算，光谱生成
post_fp函数：根据任务不同，处理输出文件，生成最终结果，可能是json文件，可能是光谱数据
'''

path_prefix=Path('/vepfs/fs_users/chensq/project/funmg/runtime_data/tasks/dft')


def gaussian(x, mu, fwhm, amplitude):
    sigma = fwhm /(2 * np.sqrt(2 * np.log(2)))  # 展宽系数
    return 28700 * amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))



def run_fp(params:Params, path_prefix:Path=path_prefix)->Any:
    # 运行可能的任务
    if params.title == 's0':
        try:
            params_cp = copy.deepcopy(params)
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix)
            g16_job.run()
        except Exception as e:
            logging.error(f'Error running s0: {e}')

    elif params.title == 's1':
        try:
            # 运行s0基态结构优化计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 's0'
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params, path_prefix)
            g16_job.run()

            # 运行s0下的td单点计算
            params_cp = copy.deepcopy(params)
            params_cp.td = 30
            params_cp.title = 's0_td_single'
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='s0.fchk')
            g16_job.run()

            # 运行s1结构优化
            params_cp = copy.deepcopy(params)
            params_cp.opt = True
            params_cp.td = 30
            params_cp.title = 's1'
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix,fchk_name='s0.fchk')
            g16_job.run()

            # 运行s1态结构优化极小点下，运行s0的单点计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 's1_ground_single'
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='s1.fchk')
            g16_job.run()
        except Exception as e:
            logging.error(f'Error running s1: {e}')

    elif params.title == 't1':
        try:
            # 运行s0基态结构优化计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 's0'
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix)
            g16_job.run()
            
            # s0极小点结构坐标下，计算t1态
            params_cp = copy.deepcopy(params)
            params_cp.title = 't1_td_singlet'
            params_cp.multiplicity = 3 
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='s0.fchk')
            g16_job.run()

            # t1态下结构优化
            params_cp = copy.deepcopy(params)
            params_cp.title = 't1'
            params_cp.multiplicity = 3 
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='s0.fchk')
            g16_job.run()

            # t1态结构优化极小点下，运行s0单点计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 't1_ground_single'
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='t1.fchk')
            g16_job.run()
        except Exception as e:
            logging.error(f'Error running t1: {e}')

    elif params.title == 'absorption_spec':
        try:
            # 运行s0基态结构优化计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 's0'
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix)
            g16_job.run()
            
            # s1态的单点计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 's1_single'
            params_cp.td = 30
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='s0.fchk')
            g16_job.run()
        except Exception as e:
            logging.error(f'Error running absorption_spec: {e}')

    elif params.title == 'emission_spec':
        try:
            # 等同于计算s1          comment: 等吗??
            params_cp = copy.deepcopy(params)
            params_cp.title = 's0'
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix)
            g16_job.run()

            # 生成s1态的单点计算
            params_cp = copy.deepcopy(params)
            params_cp.title = 's1'
            params_cp.td = 10
            params_cp.opt = True
            g16_job=G16Input.gen_input_file_and_return_self(params_cp, path_prefix, fchk_name='s0.fchk')
            g16_job.run()

        except Exception as e:
            logging.error(f'Error running emission_spec: {e}')

    else:
        logging.error(f'Unsupported task: {params.title}')
    return 

def post_fp(params:Params, path_prefix:Path=path_prefix)->Any:
    '''
    后处理运行任务, 生成json文件或者光谱数据 
    '''
    task_name = params.title

    data = {}
    if task_name == 's0':
        try:
            # 后处理s0, 如果是s0, 只有基态能量相对有用
            with open(Path(path_prefix, 's0.fchk'), 'r') as f:
                # 由于我们只采用dft方法，所以s0能量就是scf能量
                energy_s0 = fchic.deck_load(f, "SCF Energy")
            data['e_s0'] = energy_s0
            converge = True
            with open(Path(path_prefix,'s0.log'),'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    converge = True
                else:
                    converge = False
            data['converge'] = converge
            with open(Path(path_prefix,'data.json'),'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post s0: {e}')
    
    elif task_name == 's1':
        try:
            converge = True 
            # 后处理s0 s1, 重要信息包括发射能，重组能 
            with open(Path(path_prefix,'s1.fchk'),'r') as f:
                # 读取优化后第一激发态能量
                td_s1 = np.array(fchic.deck_load(f, "CIS Energy"))
            with open(Path(path_prefix,'s1.log'),'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'s1_ground_single.fchk'), 'r') as f:
                # 读取优化后结构的基态能量
                ground_s1 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open(Path(path_prefix,'s1_ground_single.log'),'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'s0.fchk'), 'r') as f:
                # 读取基态能量
                ground_s0 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open(Path(path_prefix,'s0.log'), 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'s0_td_single.fchk'), 'r') as f:
                # 读取s0的td能量
                td_s0 = np.array(fchic.deck_load(f, "CIS Energy"))
            with open(Path(path_prefix,'s0_td_single.log'), 'r') as f:
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
            with open(Path(path_prefix,'data.json'),'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post s1: {e}')
    
    elif task_name == 't1':
        try:
            converge = True
            # 后处理t1, 重要信息包括发射能，重组能
            with open(Path(path_prefix,'t1.fchk'), 'r') as f:
                td_t1 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open(Path(path_prefix,'t1.log'), 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'t1_ground_single.fchk'), 'r') as f:
                ground_t1 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open(Path(path_prefix,'t1_ground_single.log'), 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'s0.fchk'), 'r') as f:
                ground_s0 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open(Path(path_prefix,'s0.log'), 'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'t1_td_singlet.fchk'), 'r') as f:
                td_s0 = np.array(fchic.deck_load(f, "SCF Energy"))
            with open(Path(path_prefix,'t1_td_singlet.log'), 'r') as f:
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
            with open(Path(path_prefix,'data.json'),'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post t1: {e}')

    elif task_name == 'absorption_spec':
        try:
            converge = True
            # 后处理吸收光谱，读取吸收能&edme, 然后生成光谱数据    
            e_abs = []
            fs = []
            with open(Path(path_prefix,'s1_single.log'),'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 4:
                        if line[0] == 'Excited' and line[1] == 'State':
                            e_abs.append(float(line[4]))
                            fs.append(float(line[-2].split('=')[-1]))
            with open(Path(path_prefix,'s0.log'),'r') as f:
                line = f.readlines()[-1].strip().split()
                if line[0] == 'Normal' and line[1] == 'termination':
                    pass
                else:
                    converge = False
            with open(Path(path_prefix,'s1_single.log'),'r') as f:
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
            with open(Path(path_prefix,'data.json'), 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f'Error post absorption_spec: {e}')
    
    elif task_name == 'emission_spec':
        try:
            # 后处理发射光谱，读取发射能, 直接采用高斯展宽 
            # 暂时不支持
            print('Not support emission_ele_spec')
        except Exception as e:
            logging.error(f'Error post emission_spec: {e}')
    return 


def flow(params:Params, path_prefix:Path):
    try:
        # 运行任务
        run_fp(params,path_prefix)
        # 后处理, 包括生成json文件或者光谱数据
        post_fp(params,path_prefix)
    except Exception as e:
        logging.warning(f'dft calculation warning: {e}')


def main():
    '''
    main函数需要解析params参数(json文件暂定)，调用make_fp, run_fp, post_fp函数
    json content example:
    {
        'task':'s0', 
        'molecule_filename':'input.xyz', 
        'nproc':4, 
        'mem':'2GB', 
        'method':'b3lyp', 
        'basis':'6-31g', 
        'charge':0, 
        'multiplicity':1
    }

    '''
    # 前处理, 包括检查任务是否合法，检查输入是否存在等
    params_path=Path('test','input.json')
    params = Params.parse_from_json(params_path)
    # 运行任务
    run_fp(params)
    # 后处理, 包括生成json文件或者光谱数据
    post_fp(params)
    return 

    

if __name__ == '__main__':
    main()
    

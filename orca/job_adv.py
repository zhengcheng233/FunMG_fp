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

    
def opt_calculator(args:argparse.Namespace):
    '''
    name.xyz 恰好为优化后的结构
    '''
    in_pth = Path(args.in_pth); out_pth = Path(args.out_pth); tmp_pth = Path(args.tmp_pth)
    electronic_state = args.electronic_state; func = args.method 
    if os.path.exists(f'{out_pth}/result.pkl'):
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}; result['Error'] = []  # 错误信息
    os.makedirs(tmp_pth, exist_ok=True)

    if electronic_state == 'S1':
        keywords = [f'! opt {func} d4 def2-svp def2/J RIJCOSX tightSCF CPCM(Toluene)', '%maxcore 3000', f'%pal nprocs 30 end', \
                    '%tddft', 'nroots 3', 'TDA false', 'iroot 1', 'end', '', '* xyz 0 1']
    elif electronic_state == 'S0':
        keywords = [f'! opt {func} d4 def2-svp def2/J RIJCOSX tightSCF CPCM(Toluene)', '%maxcore 3000', f'%pal nprocs 30 end', '', '* xyz 0 1']
    
    
    with open(Path(out_pth, 'result.pkl'), 'rb') as f:
        data = pickle.load(f)
        # 读取坐标和符号
        if electronic_state == 'S0':
            coord = np.array(data['coord'])
            symbol = data['symbol']
        elif electronic_state == 'S1':
            coord = np.array(data['s0opt_coord'])
            symbol = data['symbol']

    result['symbol'] = symbol
    result['converge'] = True  if 'converge' not in result.keys() else result['converge']
    if electronic_state == 'S0':
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
    electronic_state = args.electronic_state; func = args.method
    if os.path.exists(f'{out_pth}/result.pkl'):
        with open(Path(out_pth, 'result.pkl'), 'rb') as f:
            result = pickle.load(f)
    else:
        result = {}; result['Error'] = []  # 错误信息
    os.makedirs(tmp_pth, exist_ok=True)
    try:
        if electronic_state == 'S0':
            keywords = [f'! {func} d4 def2-svp def2/J RIJCOSX tightSCF CPCM(Toluene)', '%maxcore 3000', f'%pal nprocs 30 end', '', '* xyz 0 1']
            coord = result['s0opt_coord']; symbol = result['symbol']
            if 'homo' in result.keys():
                return 
            geometry_2_input.geom_2_inp(Path(tmp_pth, 's0_ground.inp'), coord, symbol, keywords)
        elif electronic_state == 'S0_td':
            keywords = [f'! {func} d4 def2-svp def2/J RIJCOSX tightSCF CPCM(Toluene)', '%maxcore 3000', f'%pal nprocs 30 end', '%tddft', 'nroots 10',\
                        'TDA false', 'end', '', '* xyz 0 1'] 
            coord = result['s0opt_coord']; symbol = result['symbol']
            if 'abs_wavelength' in result.keys():
                return 
            geometry_2_input.geom_2_inp(Path(tmp_pth, 's0_td.inp'), coord, symbol, keywords)
        elif electronic_state == 'S1_td':
            keywords= [f'! {func} d4 def2-svp def2/J RIJCOSX tightSCF CPCM(Toluene)', '%maxcore 3000', f'%pal nprocs 30 end', '%tddft', 'nroots 10',
                        'TDA false', 'end', '', '* xyz 0 1'] 
            coord = result['s1opt_coord']; symbol = result['symbol']
            if 'emi_wavelength' in result.keys():
                return
            geometry_2_input.geom_2_inp(Path(tmp_pth, 's1_td.inp'), coord, symbol, keywords)

        if electronic_state == 'S0':
            with open(Path(tmp_pth, 's0_ground_err.out'), 'w') as f1, open(Path(tmp_pth, 's0_ground_out.out'), 'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's0_ground.inp'], check=True, stdout=f2, stderr=f1)
            with open(Path(tmp_pth, 's0_ground_out.out'), 'r') as f:
                lines = f.readlines(); read_data = False; homo = []; lumo = []
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
            if len(homo) > 0:
                result['homo'] = homo[-1]; result['lumo'] = lumo[0]
        if electronic_state == 'S0_td':
            with open(Path(tmp_pth, 's0_td_err.out'),'w') as f1, open(Path(tmp_pth,'s0_td_out.out'),'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's0_td.inp'], check=True, stdout=f2, stderr=f1)
            wavelength = None; fosc= None 
            with open(Path(tmp_pth, 's0_td_out.out'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 4 and line[0] == '0-1A' and line[1] == '->' and line[2] == '1-1A' and wavelength is None:
                        wavelength = float(line[5]); fosc = float(line[6])
            if wavelength is not None and fosc is not None:
                result['abs_wavelength'] = wavelength; result['abs_fosc'] = fosc

        if electronic_state == 'S1_td':
            with open(Path(tmp_pth, 's1_td_err.out'),'w') as f1, open(Path(tmp_pth,'s1_td_out.out'),'w') as f2:
                subprocess.run(['/public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/app/orca', 's1_td.inp'], check=True, stdout=f2, stderr=f1)
            wavelength = None; fosc= None
            with open(Path(tmp_pth, 's1_td_out.out'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if len(line) > 4 and line[0] == '0-1A' and line[1] == '->' and line[2] == '1-1A' and wavelength is None:
                        wavelength = float(line[5]); fosc = float(line[6])
            if wavelength is not None and fosc is not None:
                result['emi_wavelength'] = wavelength; result['emi_fosc'] = fosc
    except:
        print(traceback.format_exc())
        result['converge'] = False  
    with open(Path(out_pth, 'result.pkl'), 'wb') as f:
        pickle.dump(result, f)
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
    
    # subparser for orca single calculator
    parser_orca_single_calculator = subparsers.add_parser('orca_single_calculator', help='Run single point calculation with ORCA')
    parser_orca_single_calculator.add_argument('--in_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Input path for com')
    parser_orca_single_calculator.add_argument('--out_pth', type=str, default='/public/home/chengz/FunMG/frame_0', help='Output path for optimization results')
    parser_orca_single_calculator.add_argument('--tmp_pth', type=str, default='/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID/frame_0', help='Temporary path for optimization')
    parser_orca_single_calculator.add_argument('--method', type=str, default='b3lyp', help='Method for optimization')
    parser_orca_single_calculator.add_argument('--electronic_state', type=str, default='S0', help='Electronic state for optimization')

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
        gen_data(args)
    
    if args.command == 'structure_opt':
        # 结构优化计算
        opt_calculator(args)
    
    if args.command == 'orca_single_calculator':
        # 单点性质计算
        orca_single_calculator(args)


    ##################################################
    # 根据需求组装单个任务,并生成相应的slurm 脚本
    ##################################################
    if args.command == 'gen_property_slurm':
        # 生成从分子结构生成-光物理性质计算脚本; 提供软件环境slurm环境；提供计算参数
        os.makedirs(args.slurm_task_pth, exist_ok=True)  # 确保任务路径存在
        gen_property_slurm(args)

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

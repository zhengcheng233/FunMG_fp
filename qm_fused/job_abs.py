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
from ase.data import chemical_symbols
from glob import glob
import lmdb
import pickle
import traceback

def abs_spectrum(args: argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); out_pth = Path(args.out_pth)
    orca_script_pth = Path(args.orca_script_pth)
    gau_script_pth = Path(args.gau_script_pth); script_pth = Path(args.script_pth)
    soft_env = args.soft_env; platform_env = args.platform_env
    nproc = args.nproc; orca_sov = args.orca_sov; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; mem = args.mem
    gau_sov = args.gau_sov; orca_method = args.orca_method; gau_method = args.gau_method
    orca_method = orca_sov + ' ' + orca_method if str(orca_sov) != '0' else orca_method
    gau_method = gau_sov + ' ' + gau_method if str(gau_sov) != '0' else gau_method

    calc_files = glob(f'{in_pth}/*/result.pkl'); num_task = 1000 
    calc_files_set = [[] for ii in range(num_task)]
    
    for idx, f_name in enumerate(calc_files):
        calc_files_set[idx % num_task].append(f_name)

    for ii, f_names in enumerate(calc_files_set):
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend(soft_env.strip().split(';')) if soft_env else []
        for f_name in f_names:
            f_dir = os.path.dirname(f_name)
            idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
            slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
            slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}',  f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --nproc {nproc} --sov {orca_sov}', \
                            f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --nproc {nproc} --sov {orca_sov} --electronic_state Abs_base --freq_file_0 s0_opt.hess', \
                            f'rm -rf {tmp_pth}'])
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{ii}.slurm')
    return 

def add_common_args(parser):
    pth = str(Path(__file__).resolve().parent.parent)
    orca_pth = f'{pth}/orca/job_adv.py';  g16_pth = f'{pth}/g16/job_adv.py'
    parser.add_argument('--in_pth', type=str, required=True, help='Input path for the molecules')
    parser.add_argument('--slurm_task_pth', type=str, required=True, help='Path to save slurm tasks')
    parser.add_argument('--tmp_pth', type=str, required=True, help='Temporary path for calculations')
    parser.add_argument('--out_pth', type=str, required=True, help='Output path for results')
    parser.add_argument('--orca_script_pth', type=str, default='', help='Path to ORCA script')
    parser.add_argument('--gau_script_pth', type=str, default='', help='Path to Gaussian script')
    parser.add_argument('--script_pth', type=str, default='', help='Path to the script')
    parser.add_argument('--soft_env', type=str, default=None, help='Software environment setup commands')
    parser.add_argument('--platform_env', type=str, default=None, help='Platform environment setup commands')
    parser.add_argument('--nproc', type=int, default=16, help='Number of processors to use')
    parser.add_argument('--orca_sov', type=str, default='RIJCOSX', help='ORCA SOV setting')
    parser.add_argument('--basis', type=str, default='def2-SVP', help='Basis set for Gaussian calculations')
    parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    parser.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity')
    parser.add_argument('--freq', type=str, default='False', help='Frequency calculation setting')
    parser.add_argument('--mem', type=str, default='50GB', help='Memory allocation for Gaussian calculations')
    parser.add_argument('--gau_sov', type=str, required=True, help='Gaussian SOV setting')
    parser.add_argument('--orca_method', type=str, default='PBE0', help='ORCA method for calculations')
    parser.add_argument('--gau_method', type=str, default='B3LYP', help='Gaussian method for calculations')


def main():
    # 吸收光谱数据集
    
    # 两个任务， 一个hessian矩阵转换问题， 另一个是光谱计算问题 
    parser = argparse.ArgumentParser(description='FunMG Spectrum Calculation')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    spectrum = subparsers.add_parser('spectrum', help='Spectrum calculation')
    add_common_args(spectrum)

    args = parser.parse_args()

    if args.command == 'spectrum':
        # 处理光谱计算任务
        abs_spectrum(args)

if __name__ == '__main__':
    main()
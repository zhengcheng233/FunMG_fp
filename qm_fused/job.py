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

# 计算吸收光谱 & 发射光谱 需要gaussian基于orca优化的结构继续优化结构获取freq，然后orca计算光谱
def fluroe_spectrum_slurm(args: argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); out_pth = Path(args.out_pth)
    orca_script_pth = Path(args.orca_script_pth)
    gau_script_pth = Path(args.gau_script_pth); script_pth = Path(args.script_pth)
    soft_env = args.soft_env; platform_env = args.platform_env
    nproc = args.nproc; orca_sov = args.orca_sov; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; mem = args.mem; method = args.method
    gau_sov = args.gau_sov; orca_method = args.orca_method; gau_method = args.gau_method
    orca_method = orca_sov + ' ' + orca_method if str(orca_sov) == '0' else orca_method
    gau_method = gau_sov + ' ' + gau_method if str(gau_sov) == '0' else gau_method

    calc_files = glob(f'{in_pth}/*/result.pkl')
    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)
        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(soft_env.strip().split(';')) if soft_env else []
        # 混合计算，orca负责结构优化，gaussian 负责频率计算， orca负责光谱计算, 高斯hessian转orca, 同时生成orca数据集，需要计算两个极小点 
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}', f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --nproc {nproc} --sov {orca_sov}', \
           f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S1 --nproc {nproc} --sov {orca_sov}', \
           f'python {gau_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S0', \
           f'python {gau_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S1', \
           f'python {orca_script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --sov {orca_sov} --nproc {nproc}', \
           f'python {orca_script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S1 --sov {orca_sov} --nproc {nproc}', \
           f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state Abs --method {orca_method} --sov {orca_sov} --freq_file_0 s0_opt.hess --freq_file_1 s1_opt.hess --nproc {nproc}', \
           f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state Emi_fluor --method {orca_method} --sov {orca_sov} --freq_file_0 s0_opt.hess --freq_file_1 s1_opt.hess --nproc {nproc}', 
           f'cp -r {tmp_pth} /public/home/chengz/photomat/fs_projects/mrtadf'
           f'rm -rf {tmp_pth}'
        ])
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def fluroe_and_kr_slurm(args: argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); out_pth = Path(args.out_pth)
    orca_script_pth = Path(args.orca_script_pth)
    gau_script_pth = Path(args.gau_script_pth); script_pth = Path(args.script_pth)
    soft_env = args.soft_env; platform_env = args.platform_env
    nproc = args.nproc; orca_sov = args.orca_sov; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; mem = args.mem 
    gau_sov = args.gau_sov; orca_method = args.orca_method; gau_method = args.gau_method
    sov = orca_sov
    calc_files = glob(f'{in_pth}/*/result.pkl')
    method = args.method

    orca_method = orca_sov + ' ' + orca_method if str(orca_sov) != '0' else orca_method
    gau_method = gau_sov + ' ' + gau_method  if str(gau_sov) != '0' else gau_method

    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)
        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(soft_env.strip().split(';')) if soft_env else []
        
        # 混合计算，orca负责结构优化，gaussian 负责频率计算， orca负责光谱计算, 高斯hessian转orca, 同时生成orca数据集，需要计算两个极小点 
        # PBE0; REVPBE38
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}', f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --nproc {nproc} --sov {sov}', \
           f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S1 --nproc {nproc} --sov {sov}', \
           f'python {gau_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method {gau_method} --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S0 --sov "scrf=solvent=Toluene"', \
           f'python {gau_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method {gau_method} --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S1 --sov "scrf=solvent=Toluene"', \
           f'python {orca_script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "CPCM(Toluene)" REVPBE38 --electronic_state S0 --sov {sov} --nproc {nproc}', \
           f'python {orca_script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method "CPCM(Toluene)" PRVPBE38 --electronic_state S1 --sov {sov} --nproc {nproc}', \
           f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state Abs --method {orca_method} --sov {sov} --freq_file_0 s0_opt.hess --freq_file_1 s1_opt.hess --nproc {nproc}', \
           f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state Emi_fluor --method {orca_method} --sov {sov} --freq_file_0 s0_opt.hess --freq_file_1 s1_opt.hess --nproc {nproc}', \
           f'python {orca_script_pth}  ic_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state None --method {orca_method} --sov {sov} --freq_file_0 s0_opt.hess --freq_file_1 s1_opt.hess --nproc {nproc}', \
           f'rm -rf {tmp_pth}'
        ])
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 


def phosphor_spectrum_slurm(args: argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); out_pth = Path(args.out_pth)
    orca_script_pth = Path(args.orca_script_pth)
    gau_script_pth = Path(args.gau_script_pth); script_pth = Path(args.script_pth)
    soft_env = args.soft_env; platform_env = args.platform_env
    nproc = args.nproc; orca_sov = args.orca_sov; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; mem = args.mem 
    calc_files = glob(f'{in_pth}/*/result.pkl')
    gau_sov = args.gau_sov; orca_method = args.orca_method; gau_method = args.gau_method
    sov = orca_sov

    orca_method = orca_sov + ' ' + orca_method if str(orca_sov) == '0' else orca_method
    gau_method = gau_sov + ' ' + gau_method if str(gau_sov) == '0' else gau_method

    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)
        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(soft_env.strip().split(';')) if soft_env else []
        # 混合计算，orca负责结构优化，gaussian 负责频率计算， orca负责光谱计算, 高斯hessian转orca, 同时生成orca数据集
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}', f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --nproc {nproc} --sov {sov}', \
           f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state T1_uhf --nproc {nproc} --sov {sov}', \
           f'python {gau_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --freq {freq} --electronic_state S0', \
           f'python {orca_script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --sov {sov} --nproc {nproc}', \
           f'python {orca_script_pth} orca_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state T1_uhf --sov {sov} --nproc {nproc}', \
           f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state Emi_phosphor --method {orca_method} --sov {sov} --freq_file_0 s0_opt.hess --freq_file_1 t1_opt.hess --nproc {nproc}', \
           f'rm -rf {tmp_pth}'
       ])
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def phosphor_slurm(args: argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); out_pth = Path(args.out_pth)
    orca_script_pth = Path(args.orca_script_pth)
    gau_script_pth = Path(args.gau_script_pth); script_pth = Path(args.script_pth)
    soft_env = args.soft_env; platform_env = args.platform_env
    nproc = args.nproc; orca_sov = args.orca_sov; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; mem = args.mem 
    calc_files = glob(f'{in_pth}/*/result.pkl')
    gau_sov = args.gau_sov; orca_method = args.orca_method; gau_method = args.gau_method

    orca_method = orca_sov + ' ' + orca_method if str(orca_sov) == '0' else orca_method
    gau_method = gau_sov + ' ' + gau_method if str(gau_sov) == '0' else gau_method

    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)

        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(soft_env.strip().split(';')) if soft_env else []
        # 混合计算，orca负责结构优化， orca负责光谱计算, 高斯hessian转orca, 同时生成orca数据集
    
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}', f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0 --nproc {nproc} --sov {orca_sov}',\
            f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state T1 --nproc {nproc} --sov {orca_sov}',\
                f'python {gau_script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity}  --electronic_state S0 --sov {gau_sov}',\
                    f'python {gau_script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state T1 --sov {gau_sov}',\
                            f'python {orca_script_pth} soc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state None --method {orca_method} --sov {orca_sov} --nproc {nproc}',\
                                        f'rm -rf {tmp_pth}'])
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def phosphor_and_plqy_slurm(args: argparse.Namespace):
    in_pth = Path(args.in_pth); slurm_task_path = Path(args.slurm_task_pth)
    tmp_pth = Path(args.tmp_pth); out_pth = Path(args.out_pth)
    orca_script_pth = Path(args.orca_script_pth)
    gau_script_pth = Path(args.gau_script_pth); script_pth = Path(args.script_pth)
    soft_env = args.soft_env; platform_env = args.platform_env
    nproc = args.nproc; orca_sov = args.orca_sov; basis = args.basis; charge = args.charge; multiplicity = args.multiplicity
    freq = args.freq; mem = args.mem 
    calc_files = glob(f'{in_pth}/*/result.pkl')
    gau_sov = args.gau_sov; orca_method = args.orca_method; gau_method = args.gau_method

    orca_method = orca_sov + ' ' + orca_method if str(orca_sov) == '0' else orca_method
    gau_method = gau_sov + ' ' + gau_method if str(gau_sov) == '0' else gau_method

    for idx, f_name in enumerate(calc_files):
        f_dir = os.path.dirname(f_name)

        idx = int(os.path.basename(f_dir).split('_')[-1])  # 获取文件夹名称
        slurm_txt = ['#!/bin/bash', f'{platform_env}', f'#SBATCH -J FunMG_{idx}', '#SBATCH -N 1',\
                    f'#SBATCH -n {nproc}']
        slurm_txt.extend([f'mkdir -p {tmp_pth}',f'mkdir -p {tmp_pth}/mol_{idx}'])
        slurm_txt.extend(soft_env.strip().split(';')) if soft_env else []
        # 混合计算，orca负责结构优化， orca负责光谱计算, 高斯hessian转orca, 同时生成orca数据集
    
        slurm_txt.extend([f'cd {tmp_pth}/mol_{idx}', f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state S0_freq --nproc {nproc} --sov {orca_sov}',\
            f'python {orca_script_pth} structure_opt --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --method {orca_method} --electronic_state T1_uhf --nproc {nproc} --sov {orca_sov}',\
                f'python {gau_script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity}  --electronic_state S0 --sov {gau_sov}',\
                    f'python {gau_script_pth} gau_single_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --nproc {nproc} --memory {mem} --method "{gau_method}" --basis "{basis}" --charge {charge} --multiplicity {multiplicity} --electronic_state T1 --sov {gau_sov}',\
                        f'python {orca_script_pth} spectrum_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state Emi_phosphor --method {orca_method} --sov {orca_sov} --freq_file_0 s0_opt.hess --freq_file_1 t1_opt.hess --nproc {nproc}',\
                            f'python {orca_script_pth} soc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state None --method {orca_method} --sov {orca_sov} --nproc {nproc}',\
                                f'python {orca_script_pth} isc_calculator --in_pth {f_dir} --out_pth {f_dir} --tmp_pth {tmp_pth}/mol_{idx} --electronic_state None --method {orca_method} --sov {orca_sov} --freq_file_0 s0_opt.hess --freq_file_1 t1_opt.hess --nproc {nproc}',\
                                    f'cp -r {tmp_pth} /public/home/chengz/photomat/fs_projects/shangjiao_orca',\
                                        f'rm -rf {tmp_pth}'])
        slurm.genslurm(slurm_txt, f'{slurm_task_path}/FunMG_{idx}.slurm')
    return 

def add_common_args(parser):
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
    pth = str(Path(__file__).resolve().parent.parent)
    orca_pth = f'{pth}/orca/job_adv.py';  g16_pth = f'{pth}/g16/job_adv.py'
    # 两个任务， 一个hessian矩阵转换问题， 另一个是光谱计算问题 
    parser = argparse.ArgumentParser(description='FunMG Spectrum Calculation')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 荧光光谱 & ic计算
    fluroe_and_kr_slurm_parser = subparsers.add_parser('fluroe_and_kr_slurm', help='Fluorescence and IC spectrum calculation')
    fluroe_and_kr_slurm_parser.add_argument('--in_pth', type=str, required=True, help='Input path for the molecules')
    fluroe_and_kr_slurm_parser.add_argument('--slurm_task_pth', type=str, required=True, help='Path to save slurm tasks')
    fluroe_and_kr_slurm_parser.add_argument('--tmp_pth', type=str, required=True, help='Temporary path for calculations')
    fluroe_and_kr_slurm_parser.add_argument('--out_pth', type=str, required=True, help='Output path for results')
    fluroe_and_kr_slurm_parser.add_argument('--orca_script_pth', type=str, default=orca_pth, help='Path to ORCA script')
    fluroe_and_kr_slurm_parser.add_argument('--gau_script_pth', type=str, default=g16_pth, help='Path to Gaussian script')
    fluroe_and_kr_slurm_parser.add_argument('--script_pth', type=str, default=pth, help='Path to the script')
    fluroe_and_kr_slurm_parser.add_argument('--soft_env', type=str, default=None, help='Software environment setup commands')
    fluroe_and_kr_slurm_parser.add_argument('--platform_env', type=str, default=None, help='Platform environment setup commands')
    fluroe_and_kr_slurm_parser.add_argument('--nproc', type=int, default=16, help='Number of processors to use')
    fluroe_and_kr_slurm_parser.add_argument('--orca_sov', type=str, default='RIJCOSX', help='ORCA SOV setting')
    fluroe_and_kr_slurm_parser.add_argument('--basis', type=str, default='def2-SVP', help='Basis set for Gaussian calculations')
    fluroe_and_kr_slurm_parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    fluroe_and_kr_slurm_parser.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity')
    fluroe_and_kr_slurm_parser.add_argument('--freq', type=str, default='False', help='Frequency calculation setting')
    fluroe_and_kr_slurm_parser.add_argument('--mem', type=str, default='50GB', help='Memory allocation for Gaussian calculations')
    fluroe_and_kr_slurm_parser.add_argument('--method', type=str, default='PBE0', help='Method for calculations')
    fluroe_and_kr_slurm_parser.add_argument('--gau_sov', type=str, default='scrf=solvent=Toluene', help='Gaussian SOV setting')
    fluroe_and_kr_slurm_parser.add_argument('--orca_method', type=str, default='PBE0', help='ORCA method for calculations')
    fluroe_and_kr_slurm_parser.add_argument('--gau_method', type=str, default='B3LYP', help='Gaussian method for calculations')
    
    # 荧光光谱计算
    fluroe_slurm_parser = subparsers.add_parser('fluroe_slurm', help='Fluorescence and IC spectrum calculation')
    fluroe_slurm_parser.add_argument('--in_pth', type=str, required=True, help='Input path for the molecules')
    fluroe_slurm_parser.add_argument('--slurm_task_pth', type=str, required=True, help='Path to save slurm tasks')
    fluroe_slurm_parser.add_argument('--tmp_pth', type=str, required=True, help='Temporary path for calculations')
    fluroe_slurm_parser.add_argument('--out_pth', type=str, required=True, help='Output path for results')
    fluroe_slurm_parser.add_argument('--orca_script_pth', type=str, default=orca_pth, help='Path to ORCA script')
    fluroe_slurm_parser.add_argument('--gau_script_pth', type=str, default=g16_pth, help='Path to Gaussian script')
    fluroe_slurm_parser.add_argument('--script_pth', type=str, default=pth, help='Path to the script')
    fluroe_slurm_parser.add_argument('--soft_env', type=str, default=None, help='Software environment setup commands')
    fluroe_slurm_parser.add_argument('--platform_env', type=str, default=None, help='Platform environment setup commands')
    fluroe_slurm_parser.add_argument('--nproc', type=int, default=16, help='Number of processors to use')
    fluroe_slurm_parser.add_argument('--orca_sov', type=str, default='RIJCOSX', help='ORCA SOV setting')
    fluroe_slurm_parser.add_argument('--basis', type=str, default='def2-SVP', help='Basis set for Gaussian calculations')
    fluroe_slurm_parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    fluroe_slurm_parser.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity')
    fluroe_slurm_parser.add_argument('--freq', type=str, default='False', help='Frequency calculation setting')
    fluroe_slurm_parser.add_argument('--mem', type=str, default='50GB', help='Memory allocation for Gaussian calculations')
    fluroe_slurm_parser.add_argument('--method', type=str, default='PBE0', help='Method for calculations')
    fluroe_slurm_parser.add_argument('--gau_sov', type=str, default='scrf=solvent=Toluene', help='Gaussian SOV setting')
    fluroe_slurm_parser.add_argument('--orca_method', type=str, default='PBE0', help='ORCA method for calculations')
    fluroe_slurm_parser.add_argument('--gau_method', type=str, default='B3LYP', help='Gaussian method for calculations')

    
    # 磷光光谱(吸收&发射) & plqy计算   
    phosphor_and_plqy_slurm_parser = subparsers.add_parser('phosphor_and_plqy_slurm', help='Phosphorescence and PLQY spectrum calculation')
    phosphor_and_plqy_slurm_parser.add_argument('--in_pth', type=str, required=True, help='Input path for the molecules')
    phosphor_and_plqy_slurm_parser.add_argument('--slurm_task_pth', type=str, required=True, help='Path to save slurm tasks')
    phosphor_and_plqy_slurm_parser.add_argument('--tmp_pth', type=str, required=True, help='Temporary path for calculations')
    phosphor_and_plqy_slurm_parser.add_argument('--out_pth', type=str, required=True, help='Output path for results')
    phosphor_and_plqy_slurm_parser.add_argument('--orca_script_pth', type=str, default=orca_pth, help='Path to ORCA script')
    phosphor_and_plqy_slurm_parser.add_argument('--gau_script_pth', type=str, default=g16_pth, help='Path to Gaussian script')
    phosphor_and_plqy_slurm_parser.add_argument('--script_pth', type=str, default=pth, help='Path to the script')
    phosphor_and_plqy_slurm_parser.add_argument('--soft_env', type=str, default=None, help='Software environment setup commands')
    phosphor_and_plqy_slurm_parser.add_argument('--platform_env', type=str, default=None, help='Platform environment setup commands')
    phosphor_and_plqy_slurm_parser.add_argument('--nproc', type=int, default=16, help='Number of processors to use')
    phosphor_and_plqy_slurm_parser.add_argument('--orca_sov', type=str, default='RIJCOSX', help='ORCA SOV setting')
    phosphor_and_plqy_slurm_parser.add_argument('--basis', type=str, default='def2-SVP', help='Basis set for Gaussian calculations')
    phosphor_and_plqy_slurm_parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    phosphor_and_plqy_slurm_parser.add_argument('--multiplicity', type=int, default=1, help='Spin multiplicity')
    phosphor_and_plqy_slurm_parser.add_argument('--freq', type=str, default='False', help='Frequency calculation setting')
    phosphor_and_plqy_slurm_parser.add_argument('--mem', type=str, default='50GB', help='Memory allocation for Gaussian calculations')
    phosphor_and_plqy_slurm_parser.add_argument('--gau_sov', type=str, required=True, help='Gaussian SOV setting')
    phosphor_and_plqy_slurm_parser.add_argument('--orca_method', type=str, default='PBE0', help='ORCA method for calculations')
    phosphor_and_plqy_slurm_parser.add_argument('--gau_method', type=str, default='B3LYP', help='Gaussian method for calculations')

    phosphor_spec = subparsers.add_parser('phosphor_slurm', help='generate phosphor spectrum')
    add_common_args(phosphor_spec)
    
    args = parser.parse_args()

    if args.command == 'fluroe_and_kr_slurm':
        fluroe_and_kr_slurm(args)
    elif args.command == 'phosphor_and_plqy_slurm':
        phosphor_and_plqy_slurm(args)
    elif args.command == 'fluroe_slurm':
        fluroe_spectrum_slurm(args)
    elif args.command == 'phosphor_slurm':
        phosphor_slurm(args)

if __name__ == '__main__':
    main()

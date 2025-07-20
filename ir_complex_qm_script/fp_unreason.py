#!/usr/bin/env python
import os
import json
from glob import glob 

def make_fp(files,mdata):
    # generate slurm scripts for fp calculation, including s0, s1 opt, soc calculation, momap calculation 
    # generate python script during the calc 
    # generate slurm script and input file
    nproc = mdata['fp']['nproc']
    task_num = mdata['fp']['task_num']
    machine_queue = mdata['fp']['queue']
    user_name = mdata['fp']['user_name']
    task_set = [[] for i in range(task_num)]
    _cwd = os.getcwd()
    for ii in range(0, len(files)):
        task_set[ii % task_num].append(files[ii])
    for idx, ii in enumerate(task_set):
        with open('momap_'+str(idx)+'.slurm','w') as fp:
            fp.write('#!/bin/bash'+'\n')
            fp.write('#SBATCH -p '+str(machine_queue)+'\n')
            fp.write('#SBATCH -J momap_'+str(idx)+'\n')
            fp.write('#SBATCH -N 1'+'\n')
            fp.write('#SBATCH -n '+str(nproc)+'\n')
            fp.write('#SBATCH --exclusive'+'\n')
            fp.write('mkdir -p /tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID'+'\n')
            #fp.write('source /public/home/chengz/MOMAP-2022A/env.sh'+'\n') 
            fp.write('module load apps/orca/5.0.2/hpcx-2.7.4-gcc-7.3.1'+'\n') 
            # cp primary input file to scratch file
            for jj in ii:
                file_name = os.path.basename(jj)
                abs_path = os.path.abspath(jj)[:-len(file_name)]
                tmp_path = '/tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID/'+jj[2:-len(file_name)]
                dir_path = tmp_path 
                fp.write('mkdir -p ' + dir_path + '\n')
                fp.write('cp '+abs_path+'input.com '+ tmp_path+'\n')
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/input_gen.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/read_data.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/run_evc.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/run_kr.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/run_knr.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n')
                fp.write('cp '+_cwd+'/read_rate.py '+ tmp_path+'\n') 
                fp.write('sleep 0.5'+'\n') 
            # do qm calculation in computational node 
            #for jj in ii:
                # need s0_opt t1_opt s0_tda t1_tda soc
                 
                # s0_opt
                file_name = os.path.basename(jj)
                fp.write('cd /tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID'+'\n')
                fp.write('cd '+str(jj[:-len(file_name)])+'\n')
                fp.write('python input_gen.py s0-opt'+'\n')
                fp.write('g16 s0-opt.com'+'\n')
                fp.write('formchk s0-opt.chk s0-opt.fchk'+'\n')
                
                # t1_opt and s0_tda
                fp.write('python input_gen.py t1-opt'+'\n')
                fp.write('g16 t1-opt.com'+'\n')
                fp.write('formchk t1-opt.chk t1-opt.fchk'+'\n')
                fp.write('python input_gen.py s0-tda'+'\n')
                fp.write('g16 s0-tda.com'+'\n')

                # t1_tda soc
                fp.write('python input_gen.py t1-tda'+'\n')
                fp.write('g16 t1-tda.com'+'\n')

                fp.write('python input_gen.py soc'+'\n')
                fp.write('/public/software/apps/orca/5.0.2/hpcx-2.7.4-gcc-7.3.1/orca soc.inp > soc.out'+'\n')
            fp.write('module unload apps/orca/5.0.2/hpcx-2.7.4-gcc-7.3.1'+'\n')
            fp.write('source /public/home/chengz/MOMAP-2022A/env.sh'+'\n')
            for jj in ii:
                file_name = os.path.basename(jj)
                abs_path = os.path.abspath(jj)[:-len(file_name)]
                tmp_path = '/tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID/'+jj[2:-len(file_name)]
                fp.write('cd /tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID'+'\n')
                # collect qm data 
                fp.write('cd '+str(jj[:-len(file_name)])+'\n')
                fp.write('python read_data.py'+'\n') 
                 
                # do momap calculation             
                fp.write('python run_evc.py '+ str(nproc) + '\n')
                fp.write('python run_kr.py '+ str(nproc) + '\n')
                fp.write('python run_knr.py '+ str(nproc) + '\n')
                fp.write('python read_rate.py'+'\n')          
            
                fp.write('cd /tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID'+'\n')
            # cp results from scratch to submit dir
            #for jj in ii:
                file_name = os.path.basename(jj) 
                abs_path = os.path.abspath(jj)[:-len(file_name)]
                tmp_path = '/tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID/'+jj[2:-len(file_name)]
                fp.write('cp -r '+tmp_path+' ' + abs_path+'\n')
                fp.write('sleep 0.5'+'\n')
                fp.write('rm -rf '+tmp_path+'\n')
            fp.write('rm -rf '+'/tmp/scratch/'+str(user_name)+'/momap.$SLURM_JOB_ID'+'\n')
    return task_set 
        
def calc_fp(task_set):
    for idx in range(len(task_set)):
        os.system('sbatch momap_'+str(idx)+'.slurm')
    return 

def post_fp(files):
    pass
    return 

if __name__ == '__main__':
    #files = glob('./iter.init/02.fp/conf.1_1_1/input.com')
    files = glob('./iter.init/02.fp/conf.*/input.com')[710:]
    with open('machine.json','r') as fp:
        mdata = json.load(fp)
    task_set = make_fp(files,mdata) 
    #calc_fp(task_set)

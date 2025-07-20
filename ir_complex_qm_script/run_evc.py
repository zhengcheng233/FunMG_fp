#!/usr/bin/env python
import os
import sys

cwd_ = os.getcwd()
nproc = sys.argv[1]
os.system('mkdir -p evc')
os.system('cp s0-opt.log evc/s0-opt.log')
os.system('cp s0-opt.fchk evc/s0-opt.fchk')
os.system('cp t1-opt.log evc/t1-opt.log')
os.system('cp t1-opt.fchk evc/t1-opt.fchk')
with open('./evc/momap.inp','w') as fp:
    fp.write('do_evc=1'+'\n')
    fp.write('\n'); fp.write('&evc'+'\n')
    fp.write(' ffreq(1) = \"s0-opt.log\"'+'\n')
    fp.write(' ffreq(2) = \"t1-opt.log\"'+'\n')
    fp.write(' proj_reorg = .t.'+ '\n')
    fp.write('/'+'\n')
os.chdir('evc')
os.system('momap -i momap.inp -n '+nproc)
os.chdir(cwd_)


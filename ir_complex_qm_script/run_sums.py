import os
import json 
import sys

cwd_ = os.getcwd()
nproc = sys.argv[1]
with open('data.json','r') as fp:
    data = json.load(fp)['1.0']

def read_reorganize_e(f_name):
    do_momap = True
    with open(f_name) as fp:
        for line in fp:
            line = line.strip().split()
            if len(line) > 3:
                if line[0] == 'Total' and line[1] == 'reorganization' and line[2] == 'energy':
                    if float(line[-1]) > 10000 or float(line[-2]) > 10000:
                        do_momap = False
    return do_momap

Ead = data[0]; EDME = data[2]
os.system('mkdir -p sum')

run_dint = ['do_spec_sums = 1', '\n', '&spec_sums', 'DSFile = "evc.dint.dat"', f'Ead={Ead} au', f'dipole_abs={EDME} debye', f'dipole_emi={EDME} debye',\
            'FWHM = 200 cm-1', 'maxvib = 10', 'if_cal_ic = .f.', 'promode = 20', 'FC_eps_abs = 0.05', 'FC_eps_emi = 0.05','FC_eps_ic = 0.05',\
            'FreqScale = 1.0', 'FreqEPS = 0.01', 'Seps = 0.01', 'eps = 0.00', 'debug = .t.', 'blocksize = 1000', 'testpoints = 1000', 'TEST = .f.',\
            'flog = "spec.sums.log"', 'reduce_eps = 0.001', '/']

run_dcart = ['do_spec_sums = 1', '\n', '&spec_sums', 'DSFile = "evc.cart.dat"', f'Ead={Ead} au', f'dipole_abs={EDME} debye', f'dipole_emi={EDME} debye',\
            'FWHM = 200 cm-1', 'maxvib = 10', 'if_cal_ic = .f.', 'promode = 20', 'FC_eps_abs = 0.05', 'FC_eps_emi = 0.05','FC_eps_ic = 0.05',\
            'FreqScale = 1.0', 'FreqEPS = 0.01', 'Seps = 0.01', 'eps = 0.00', 'debug = .t.', 'blocksize = 1000', 'testpoints = 1000', 'TEST = .f.',\
            'flog = "spec.sums.log"', 'reduce_eps = 0.001', '/']

if os.path.isfile('evc/evc.dint.dat'):
    os.system('cp -r evc/evc.dint.dat sum')
    with open('sum/momap.inp','w') as f:
        run_sh = "\n".join(run_dint) 
        f.write('%s' % run_sh)
else:
    os.system('cp -r evc/evc.cart.dat sum')
    do_momap = read_reorganize_e('evc/evc.cart.dat')
    if do_momap == True:
        with open('sum/momap.inp','w') as f:
            run_sh = "\n".join(run_dcart)
            f.write('%s' % run_sh)

os.chdir('sum')
os.system('momap -i momap.inp -n ' + nproc)
os.chdir(cwd_)

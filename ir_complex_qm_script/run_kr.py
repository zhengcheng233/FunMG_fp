import os
import json
import sys

cwd_ = os.getcwd()
nproc = sys.argv[1]
with open("data.json","r") as fp:
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
os.system('mkdir -p kr')
if os.path.isfile('evc/evc.dint.dat'):
    os.system('cp -r evc/evc.dint.dat kr')
    with open('kr/momap.inp','w') as f:
        worksh = ["do_spec_tvcf_ft = 1\n", "do_spec_tvcf_spec = 1\n", "\n", "&spec_tvcf\n", " DUSHIN = .f.\n", " Temp = 298 K\n", " tmax = 7500 fs\n", " dt = 0.025 fs\n", f" Ead={Ead} au\n", f" EDMA={EDME} debye\n", f" EDME={EDME} debye\n", " DSFile=\"evc.dint.dat\"\n", " Emax = 0.3 au\n", " dE = 0.00001 au\n", " logFile = \"spec.tvcf.log\"\n", " FtFile = \"spec.tvcf.ft.dat\"\n", " FoFile = \"spec.tvcf.fo.dat\"\n", " FoSFile = \"spec.tvcf.spec.dat\"\n", "/"]
        f.writelines(worksh)
else:
    os.system('cp -r evc/evc.cart.dat kr')
    do_momap = read_reorganize_e('evc/evc.cart.dat')
    if do_momap == True:
        with open('kr/momap.inp','w') as f:
            worksh = ["do_spec_tvcf_ft = 1\n", "do_spec_tvcf_spec = 1\n", "\n", "&spec_tvcf\n", " DUSHIN = .f.\n", " Temp = 298 K\n", " tmax = 7500 fs\n", " dt = 0.025 fs\n", f" Ead={Ead} au\n", f" EDMA={EDME} debye\n", f" EDME={EDME} debye\n", " DSFile=\"evc.cart.dat\"\n", " Emax = 0.3 au\n", " dE = 0.00001 au\n", " logFile = \"spec.tvcf.log\"\n", " FtFile = \"spec.tvcf.ft.dat\"\n", " FoFile = \"spec.tvcf.fo.dat\"\n", " FoSFile = \"spec.tvcf.spec.dat\"\n", "/"]
            f.writelines(worksh)
    
os.chdir('kr')
os.system('momap -i momap.inp -n '+nproc)
os.chdir(cwd_)

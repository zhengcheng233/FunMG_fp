import os
import json
import sys

cwd_ = os.getcwd()
nproc = sys.argv[1]
with open("data.json","r") as fp:
    data = json.load(fp)['1.0']
Ead = data[0]; EDME = data[2]; Hso = data[1]

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

os.system('mkdir -p knr')
if os.path.isfile('evc/evc.dint.dat'):
    os.system('cp -r evc/evc.dint.dat knr')
    with open("knr/momap.inp", "w") as fa:
        worksh = ["do_isc_tvcf_ft = 1\n", "do_isc_tvcf_spec = 1\n", "\n", "&isc_tvcf\n", " DUSHIN = .f.\n", " Temp = 298 K\n", " tmax = 7500 fs\n", " dt = 0.025 fs\n", f" Ead={Ead} au\n", f" Hso={Hso} cm-1\n", " DSFile=\"evc.dint.dat\"\n", " Emax = 0.3 au\n", " dE = 0.00001 au\n", " logFile = \"isc.tvcf.log\"\n", " FtFile = \"isc.tvcf.ft.dat\"\n", "FoFile = \"isc.tvcf.fo.dat\"\n", "/"]
        fa.writelines(worksh)
else:
    os.system('cp -r evc/evc.cart.dat knr')
    do_momap = read_reorganize_e('evc/evc.cart.dat')
    if do_momap == True:
        with open("knr/momap.inp", "w") as fa:
            worksh = ["do_isc_tvcf_ft = 1\n", "do_isc_tvcf_spec = 1\n", "\n", "&isc_tvcf\n", " DUSHIN = .f.\n", " Temp = 298 K\n", " tmax = 7500 fs\n", " dt = 0.025 fs\n", f" Ead={Ead} au\n", f" Hso={Hso} cm-1\n", " DSFile=\"evc.cart.dat\"\n", " Emax = 0.3 au\n", " dE = 0.00001 au\n", " logFile = \"isc.tvcf.log\"\n", " FtFile = \"isc.tvcf.ft.dat\"\n", "FoFile = \"isc.tvcf.fo.dat\"\n", "/"]
            fa.writelines(worksh)

os.chdir('knr')
os.system('momap -i momap.inp -n '+nproc)
os.chdir(cwd_)



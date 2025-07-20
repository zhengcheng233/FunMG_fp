# -*- coding: utf-8 -*-
import logging
import os
import sys
import json 
import numpy as np

element_table = {
    'H':    1,  'He':   2,  'Li':   3,  'Be':   4,  'B':    5,
    'C':    6,  'N':    7,  'O':    8,  'F':    9,  'Ne':  10,
    'Na':  11,  'Mg':  12,  'Al':  13,  'Si':  14,  'P':   15,
    'S':   16,  'Cl':  17,  'Ar':  18,  'K':   19,  'Ca':  20,
    'Sc':  21,  'Ti':  22,  'V':   23,  'Cr':  24,  'Mn':  25,
    'Fe':  26,  'Co':  27,  'Ni':  28,  'Cu':  29,  'Zn':  30,
    'Ga':  31,  'Ge':  32,  'As':  33,  'Se':  34,  'Br':  35,
    'Kr':  36,  'Rb':  37,  'Sr':  38,  'Y':   39,  'Zr':  40,
    'Nb':  41,  'Mo':  42,  'Tc':  43,  'Ru':  44,  'Rh':  45,
    'Pd':  46,  'Ag':  47,  'Cd':  48,  'In':  49,  'Sn':  50,
    'Sb':  51,  'Te':  52,  'I':   53,  'Xe':  54,  'Cs':  55,
    'Ba':  56,  'La':  57,  'Ce':  58,  'Pr':  59,  'Nd':  60,
    'Pm':  61,  'Sm':  62,  'Eu':  63,  'Gd':  64,  'Tb':  65,
    'Dy':  66,  'Ho':  67,  'Er':  68,  'Tm':  69,  'Yb':  70,
    'Lu':  71,  'Hf':  72,  'Ta':  73,  'W':   74,  'Re':  75,
    'Os':  76,  'Ir':  77,  'Pt':  78,  'Au':  79,  'Hg':  80,
    'Tl':  81,  'Pb':  82,  'Bi':  83,  'Po':  84,  'At':  85,
    'Rn':  86,  'Fr':  87,  'Ra':  88,  'Ac':  89,  'Th':  90,
    'Pa':  91,  'U':   92,  'Np':  93,  'Pu':  94,  'Am':  95,
    'Cm':  96,  'Bk':  97,  'Cf':  98,  'Es':  99,  'Fm': 100,
    'Md': 101,  'No': 102,  'Lr': 103,  'Rf': 104,  'Db': 105,
    'Sg': 106,  'Bh': 107,  'Hs': 108,  'Mt': 109,  'Ds': 110,
    'Rg': 111,  'Cn': 112,  'Nh': 113,  'Fl': 114,  'Mc': 115,
    'Lv': 116,  'Ts': 117,  'Og': 118
}
element_table2 = {}
for ikey in element_table.keys():
    element_table2[element_table[ikey]] = ikey


def read_init_xyz(file):
    element_xyz = []
    # logger.info("Supported input filetype, gjf, com, xyz, chk")
    filename, file_extension = os.path.splitext(file)
    if file_extension in [".gjf", ".com", ".xyz"]:
        with open(file, "r") as filea:
            iatom = 1
            for icol in filea:
                icol = icol.split()
                if (len(icol) == 4):
                    if (icol[0].isalpha()):
                        try:
                            iel_xyz = [icol[0]] + [float(ix) for ix in icol[1:]]
                            element_xyz.append(iel_xyz)
                            # logger.info(f"Read coordinate of atom {iatom}, {iel_xyz}")
                            iatom = iatom + 1
                        except:
                            pass
                elif (len(icol) == 5):
                    if (icol[0].isalpha()) and icol[1] == '0':
                        try:
                            iel_xyz = [icol[0]] + [float(ix) for ix in icol[2:]]
                            element_xyz.append(iel_xyz)
                            # logger.info(f"Read coordinate of atom {iatom}, {iel_xyz}")
                            iatom = iatom + 1
                        except:
                            pass
    elif file_extension in [".log", ".out"]:
        with open(file, "r") as file:
            logs = file.readlines()
            chk_converge = logs[-1].split()
            assert ["Normal", "termination"] == chk_converge[:2], f"Not converged"
            line_idx = [idx for idx in range(len(logs))
                        if "Coordinates" in logs[idx].split(" ")][-1] + 3
            for iline in logs[line_idx:]:
                if iline.startswith(" ----"):
                    break
                else:
                    atomic_sym = element_table2[int(iline.split()[1])]
                    x, y, z = iline.split()[3:]
                    element_xyz.append([atomic_sym] +
                                       [float(x), float(y), float(z)])
    elif file_extension in [".chk"]:
        element_xyz = 'read_chk'
    return element_xyz


def make_opt_input(element_xyz, multiplicity, name, dump_dir=None, charge=0,
                   functional="EmpiricalDispersion=GD3BJ b3lyp", basis='def2svp',
                   pseudo_basis="Lanl2DZ", nproc=30, mem=60):
    assert name in ["s0-opt", "t1-opt"], "job type wrong"
    elems = set([ielems[0] for ielems in element_xyz])
    heavymetals = ["Ir", "Pt", "Os"]
    metal = set(heavymetals).intersection(elems)
    elems = list(elems - metal)
    elems = " ".join(elems)
    #assert len(metal) == 1, f"Heavy metals {heavymetals} not detected or more than 1"
    chk = name
    if dump_dir is None:
        dump_dir = "./s0-opt" if multiplicity == 1 else './t1-opt'
    keywords = f"opt freq {functional}/{basis} "
    if element_xyz == 'read_chk':
        keywords = f"opt freq {functional}/{basis} geom=chk"
    chkkeywords = f'%chk={chk}.chk'
    nprockeywords = f'%nproc={nproc}'
    memkeywords = f'%mem={mem}gb'
    titlekeywords = 'OLED project'
    chargekeywords = f'{charge} {multiplicity}'
    buff = [chkkeywords, nprockeywords, memkeywords, '# {}'.format(keywords), '', titlekeywords, '', chargekeywords]
    if element_xyz != 'read_chk':
        for ixyz in element_xyz:
            buff.append(f"{ixyz[0]} {ixyz[1]:f} {ixyz[2]:f} {ixyz[3]:f}")
    buff.append('')
    #pseudo_element = metal.pop()
    #other_element = elems
    #buff.append('%s 0' % (pseudo_element))
    #buff.append(pseudo_basis)
    #buff.append('****')
    #buff.append('%s 0' % (other_element))
    #buff.append(basis)
    #buff.append('****')
    #buff.append('')
    #buff.append('%s 0' % (pseudo_element))
    #buff.append(pseudo_basis)
    buff.append('\n')
    output = "\n".join(buff)
    if name == "s0-opt":
        with open(f"{dump_dir}/s0-opt.com", "w") as text_file:
            text_file.write("%s" % output)
    elif name == "t1-opt":
        with open(f"{dump_dir}/t1-opt.com", "w") as text_file:
            text_file.write("%s" % output)


def make_tda_input(element_xyz, name, dump_dir=None, nproc=30, mem=60, basis='def2-svp', pseudo_basis="Lanl2DZ"):
    elems = set([ielems[0] for ielems in element_xyz])
    heavymetals = ["Ir", "Pt", "Os"]
    metal = set(heavymetals).intersection(elems)
    elems = list(elems - metal)
    elems = " ".join(elems)
    #assert len(metal) == 1, f"Heavy metals {heavymetals} not detected or more than 1"
    assert name in ["s0-tda", "t1-tda", "t1-opt"], "job type wrong"
    if dump_dir is None:
        dump_dir = f"./{name}"
    chkkeywords = f'%chk={name}.chk'
    nprockeywords = f'%nprocshared={nproc}'
    memkeywords = f'%mem={mem}gb'
    titlekeywords = 'TDA'
    chargekeywords = '0 1'
    if name == 's0-tda':
        keywords = "TDA(50-50,NStates=3,Root=1) PBEpbe/def2svp IOP(3/76=1000003333) Iop(3/77=0666706667)  geom=chk"
    elif name == 't1-tda':
        keywords = "TDA(triplet,NStates=3,Root=1) PBEpbe/def2svp IOP(3/76=1000003333) Iop(3/77=0666706667) geom=chk"
    else:
        keywords = "TDA(triplet,NStates=3,Root=1) b3lyp/def2svp opt freq"
    buff = [chkkeywords, nprockeywords, memkeywords, '# {}'.format(keywords), '', titlekeywords, '', chargekeywords]
    if name == 't1-opt':
        for ixyz in element_xyz:
            ixyz = [str(u) for u in ixyz]
            buff.append(f"{ixyz[0]} {float(ixyz[1]):f} {float(ixyz[2]):f} {float(ixyz[3]):f}")
    buff.append('')
    #pseudo_element = metal.pop()
    #other_element = elems
    #buff.append('%s 0' % (pseudo_element))
    #buff.append(pseudo_basis)
    #buff.append('****')
    ##buff.append('%s 0' % (other_element))
    #buff.append(basis)
    #buff.append('****')
    #buff.append('')
    #buff.append('%s 0' % (pseudo_element))
    #buff.append(pseudo_basis)
    buff.append('\n')
    output = "\n".join(buff)
    with open(f"{dump_dir}/{name}.com", "w") as text_file:
        text_file.write("%s" % output)


def make_edme_input(element_xyz, dump_dir="edme"):
    make_opt_input(element_xyz, 3, "t1-opt", dump_dir)
    python_path = sys.executable
    os.system(f"{python_path} gjf2mol.py {dump_dir}/t1-opt.com")
    with open(f"{dump_dir}/t1-opt.mol", "r") as file:
        mols = file.readlines()
        metal_idx = [idx for idx in range(len(mols))
                     if len(set(["Ir", "Pt", "Os"]).intersection(mols[idx].strip().split(" ")))==1]
        if len(metal_idx) == 1:
            line_i = mols[metal_idx[0]-1]
            line_i = line_i.strip().split(" ")[:2]
            line_i.append("Basis=lanl2dz_ecp ECP=lanl2dz_ecp\n")
            mols[metal_idx[0]-1] = " ".join(line_i)
        else:
            # logger.error("Heavy metal Ir, Pt, Os not found")
            assert False
    with open(f"{dump_dir}/t1-opt.mol", "w") as file:
        file.writelines(mols)


def make_soc_input(element_xyz, dump_dir='soc'):
    elems = set([ielems[0] for ielems in element_xyz])
    heavymetals = ["Ir", "Pt", "Os"]
    metal = set(heavymetals).intersection(elems)
    metal = metal.pop()
    with open(f"{dump_dir}/soc.inp", "w") as file:
        inps = ["! B3LYP ZORA ZORA-def2-SVP SARC/J RI-SOMF(1X) nopop\n","%PAL NPROCS 30 END\n",
                "%tddft\n", "nroots 25\n", "dosoc true\n", "tda false\n", "end\n",
                f"%basis newgto {metal} \"SARC-ZORA-SVP\" end\n", "end\n", "* xyz 0 1\n"]
        for ixyz in element_xyz:
            inps.append(f"{ixyz[0]} {ixyz[1]:f} {ixyz[2]:f} {ixyz[3]:f}\n")
        inps.append("end")
        file.writelines(inps)

def read_freq(f_name,n_atom):
    freqs = []; images = []
    with open(f_name,'r') as fp:
        lines = fp.readlines()
        for idx,line in enumerate(lines):
            if line.startswith(' Frequencies --'):
                freq = []
                line = line.strip().split()
                image = [float(u) for u in line[2:]]
                line_tmp = lines[idx+5:idx+5+n_atom]
                for line0 in line_tmp:
                    line0 = line0.strip().split()
                    freq.append([float(u) for u in line0[2:]])
                freqs.append(freq); images.append(image)
    neg_freq = []
    for idx,image in enumerate(images):
        freq = np.array(freqs[idx])
        aa = (freq)[:,0:3]
        for jj in range(len(image)):
            if image[jj] < 0.:
               neg_freq.append(freq[:,jj*3:jj*3+3])
    return neg_freq

if __name__ == '__main__':
    qm_task = sys.argv[1]
    # using param.json to contral the dft param in the future
    #with open('param.json','r') as fp:
    #    jdata = json.load(fp)
    if qm_task == 's0-opt':
        element_xyz = read_init_xyz('input.com')    
        make_opt_input(element_xyz, 1, "s0-opt", ".")
    elif qm_task == 's0-opt_win':
        element_xyz = read_init_xyz('input.gjf')    
        make_opt_input(element_xyz, 1, "s0-opt", ".")
    elif qm_task == 't1-opt':
        element_xyz = read_init_xyz('s0-opt.log')
        os.system('cp s0-opt.chk t1-opt.chk')
        make_opt_input(element_xyz, 3, "t1-opt", ".")
        #make_tda_input(element_xyz, "t1-opt", ".")
    elif qm_task == 's0-tda':
        element_xyz = read_init_xyz('s0-opt.log')
        os.system("cp s0-opt.chk s0-tda.chk")
        make_tda_input(element_xyz, 's0-tda', '.')
    elif qm_task == 't1-tda':
        element_xyz = read_init_xyz('t1-opt.log')
        os.system("cp t1-opt.chk t1-tda.chk")
        make_tda_input(element_xyz, 't1-tda', '.')
    elif qm_task == 'edme':
        element_xyz = read_init_xyz('t1-opt.log')
        make_edme_input(element_xyz, ".")
    elif qm_task == 'soc':
        element_xyz = read_init_xyz('t1-opt.log')
        make_soc_input(element_xyz, ".")
    elif qm_task == 't1-opt-check':
        element_xyz = read_init_xyz('t1-opt.log')
        dire = read_freq('t1-opt.log', len(element_xyz))
        if len(dire) == 0:
            pass
        else:
            os.system('cp t1-opt.log imagenary.log')
            os.system('rm t1-opt.log')
            os.system('rm t1-opt.chk')
            os.system('rm t1-opt.fchk')
            element_xyz = np.array(element_xyz)
            ele_sym = element_xyz[:,0:1]; ele_crd = np.array(element_xyz[:,1:],dtype=np.float64)
            for _dire in dire:
                ele_crd += _dire * 0.01
            element_xyz = np.hstack((ele_sym,ele_crd))
            make_tda_input(element_xyz, "t1-opt", ".")
            os.system('g16 t1-opt.com')
            os.system('formchk t1-opt.chk t1-opt.fchk')

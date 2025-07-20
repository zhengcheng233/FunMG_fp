# -*- coding: utf-8 -*-
import numpy as np
import json

def read_t1_e(tda_log):
    with open(tda_log, "r") as f:
        file_lines = f.readlines()
    for iline in file_lines:
        content = iline.split(" ")
        if set(["Total", "Energy,"]).issubset(set(content)):
            content = [i_content for i_content in content if i_content != ""]
            return content[4]


def read_s0_e(tda_log):
    with open(tda_log, "r") as f:
        file_lines = f.readlines()
    for iline in file_lines:
        content = iline.split(" ")
        if set(["SCF", "Done:"]).issubset(set(content)):
            content = [i_content for i_content in content if i_content != ""]
            return content[4]


def read_soc_orca(fin):
    with open(fin) as fopen:
        lines = fopen.readlines()
        iline = [i for i in range(len(lines)) if "SOCME" in lines[i].split(" ")]
        assert len(iline) == 1
        iline = iline.pop()+5
        content = lines[iline].split(" ")
        content = [float(ic) for ic in content if ic not in ["", ",", ")", "(", ")\n"]]
        content = content[2:]
        content = [ic**2 for ic in content if ic != 0]
        #assert len(content) == 3
        return np.sqrt(sum(content) / 3)


def read_edme_orca(fin):
    with open(fin) as fopen:
        lines = fopen.readlines()
        iline = [i for i in range(len(lines)) if "CORRECTED" in lines[i].split(" ")]
        iline = [i for i in iline if "ELECTRIC" in lines[i].split(" ")]
        iline = [i for i in iline if "MOMENTS\n" in lines[i].split(" ")].pop() + 5

        T2 = 0
        for i in range(iline, iline+3):
            i_lines = lines[i].split(" ")
            T2 = T2 + float([x for x in i_lines if x!= ''][5])
    return np.sqrt(T2/3) * 2.5417

if __name__ == '__main__':
    e1 = read_t1_e('t1-tda.log'); e2 = read_s0_e('s0-tda.log')
    soc = read_soc_orca('soc.out'); edme = read_edme_orca('soc.out')
    E_ad = float(e1) - float(e2)
    all_data = {'1.0':[E_ad, soc, edme]}
    with open('data.json','w') as fp:
        json.dump(all_data, fp, indent=4)


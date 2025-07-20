#!/usr/bin/env python
import os
from glob import glob 

confs = []
with open('fail.txt','r') as fp:
    lines = fp.readlines()
    for idx,line in enumerate(lines):
        line = line.strip().split()
        if idx > 0:
            confs.append(line[0])

all_data0 = glob('./iter.*/conf.*')
all_data1 = glob('./iter.init/02.fp/conf.*')
all_data = all_data0 + all_data1

conf_candi = []
for conf in all_data:
    f_name = os.path.basename(conf)
    if f_name in confs:
        conf_candi.append(conf)
os.mkdir('fail')
for conf in conf_candi:
    os.system('cp -r '+conf+' fail')
print(conf_candi)

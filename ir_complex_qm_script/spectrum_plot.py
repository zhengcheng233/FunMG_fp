#!/usr/bin/env python
from glob import glob
import matplotlib.pyplot as plt
import os

f_names = glob('./*/sum/spec.sums.spec.dat')
cwd_ = os.getcwd()
def plot_spec(f_name):
    x = []; y = []
    with open(f_name,'r') as fp:
        lines = fp.readlines()
        for line in lines[1:]:
            line = line.strip().split()
            if line[-1] == '--':
                pass
            else:
                if float(line[-3]) > 400 and float(line[-3]) < 800:
                    x.append(float(line[-3]))
                    y.append(float(line[-1]))
    plt.plot(x,y)
    plt.savefig('spec.png')
    plt.clf()
    return 

for f_name in f_names:
    f_dir = os.path.dirname(f_name)
    os.chdir(f_dir)
    plot_spec(os.path.basename(f_name))
    os.chdir(cwd_)

#!/usr/bin/env python 
import os 

def genslurm(command, f_name):
    os.makedirs(os.path.dirname(f_name), exist_ok=True)
    with open(f_name, 'w') as f:
        f.write('\n'.join(command))

    
#!/usr/bin/env python 
'''
转换 orca与gaussian之间的hessian矩阵
'''
import numpy as np
import fchic 
from ase.data import atomic_masses, chemical_symbols, atomic_numbers 

def hessian_convert(hessian, out_pth, n_atoms, coord, symbol, multiplicity):
    #with open(in_pth, 'r') as f:
    #    hessian = fchic.deck_load(f, 'Cartesian Force Constants')
    #    hessian = np.array(hessian)
    #    hessian_2d = np.zeros((n_atoms * 3, n_atoms * 3))
    
    k = 0; hessian = np.array(hessian)


    if len(hessian) == (len(symbol) * 3):
        pass 
    else:
        hessian_2d = np.zeros((n_atoms * 3, n_atoms * 3))
        for i in range(0, n_atoms * 3):
            for j in range(0, i + 1):
                hessian_2d[i][j] = hessian[k]
                k = k + 1 
        Hess = np.where(hessian_2d, hessian_2d, hessian_2d.T)  
        # 转换为orca可以识别的格式
        n_sum = 0 
        with open(out_pth, 'w') as f:
            f.write(f'\n$orca_hessian_file\n\n$act_atom\n  0\n\n$act_coord\n  0\n\n$act_energy\n        0.000000\n$multiplicity\n  {multiplicity}\n\n$hessian\n')
            f.write(str(hessian_2d.shape[0]) + '\n')
            for i in range(0, (Hess.shape[0] * int(Hess.shape[0] / 5 + 1))):
                n_column = int(i / Hess.shape[0]) * 5
                if i % Hess.shape[0] == 0:
                    f.write((20 - len(str(n_column))) * ' ' + str(n_column))
                    f.write((18 - len(str(n_column + 1))) * ' ' + str(n_column + 1))
                    f.write((18 - len(str(n_column + 2))) * ' ' + str(n_column + 2))
                    f.write((18 - len(str(n_column + 3))) * ' ' + str(n_column + 3))
                    f.write((18 - len(str(n_column + 4))) * ' ' + str(n_column + 4))
                    f.write(8 * ' ' + '\n')
                for j in range(0, 5):
                    n_column = int(i / Hess.shape[0]) * 5 + j
                    if n_sum == Hess.shape[0] * Hess.shape[0]:
                        break
                    if j == 0:
                        f.write((4 - len(str(i % Hess.shape[0]))) * ' ' + str(i % Hess.shape[0]) + 5 * ' ')
                    if n_column < Hess.shape[0]:
                        if Hess[i % Hess.shape[0]][n_column] >= 0:
                            f.write(' ')
                            f.write(str(Hess[i % Hess.shape[0]][n_column]))
                        else:
                            f.write(str(Hess[i % Hess.shape[0]][n_column]))
                        f.write('  ')
                        n_sum = n_sum + 1
                    if j == 4:
                        f.write('\n')

            coord = np.array(coord) / 0.529177249  # Convert to Bohr
            f.write('\n\n')
            f.write('$atoms\n')
            f.write(str(n_atoms) + '\n')
            for i in range(0, n_atoms):
                f.write(symbol[i] + '    ')
                f.write(str(atomic_masses[atomic_numbers[symbol[i]]]) + '    ')
                f.write(str(coord[i][0]) + '    ')
                f.write(str(coord[i][1]) + '    ')
                f.write(str(coord[i][2]) + '\n')
            f.write('\n')
            f.write('$end\n')
    return 

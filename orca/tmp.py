#!/usr/bin/env python 

from glob import glob
import pickle 
import sys 
import os 
import traceback

opera = sys.argv[1]

if opera == 'coll':
    data_all = []

    for idx in range(0, 225500, 100):
        f_name = os.path.join(f'./qmcalc/frame_{idx}', 'result.pkl')
        if os.path.exists(f_name):
            with open(f_name, 'rb') as f_in:
                dd = pickle.load(f_in)
        else:
            dd = {}
        data_all.append(dd)
    #f_dirs = glob('./qm_calc/frame_*/result.pkl')

    #data_all = []
    #for f in f_dir
    #    with open(f, 'rb') as f_in:
    #        data = pickle.load(f_in)
    #        f_name = os.path.basename(os.path.dirname(f))
    #        data['f_name'] = f_name
    #        data_all.append(data)

    with open('data_all.pkl', 'wb') as f_out:
        pickle.dump(data_all, f_out)

elif opera == 'gen':

    with open(sys.argv[2],'rb') as fp:
        data = pickle.load(fp)
    k = 0 
    for i, frame in enumerate(data):
        try:
            os.makedirs(f'./qm_calc/frame_{i}', exist_ok=True)
            with open(f'./qm_calc/frame_{i}/result.pkl', 'wb') as f_out:
                frame_new = {}
                frame_new['symbol'] = frame['symbol']
                frame_new['coord_init'] = frame['coord_init']
                pickle.dump(frame, f_out)
        except Exception as e:
            print(i)
            print(traceback.format_exc())

elif opera == 'convert_pkl':
    f_dirs = glob('./*/input_xtb.com')
    for f_name in f_dirs:
        f_dir = os.path.dirname(f_name)
        coord = []; symbol = []
        with open(f_name, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) == 4:
                    coord.append([float(x) for x in line[1:4]])
                    symbol.append(line[0])
        _coord = []; _symbol = []
        for ii in range(len(symbol)):
            if symbol[ii] == 'Ir':
                _symbol.append(symbol[ii])
                _coord.append(coord[ii])
        for ii in range(len(symbol)):
            if symbol[ii] != 'Ir':
                _symbol.append(symbol[ii])
                _coord.append(coord[ii])
        with open(f'{f_dir}/result.pkl','wb') as fp:
            pickle.dump({'coord_init':_coord, 'symbol': _symbol},fp)




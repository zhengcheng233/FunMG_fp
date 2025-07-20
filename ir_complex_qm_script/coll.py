#!/usr/bin/env python

import numpy as np
fmt = '%06d'

coords = []; symbols = []; conf_idxs = []; plqys = []; e_ads = []; edmes = []; homos = []; lumos = []
coords1 = []; symbols1 = []; conf_idxs1 = []; plqys1 = []; e_ads1 = []; edmes1 = []; homos1 = []; lumos1 = []

init_npz = np.load('./iter.init/iter.000000.npz',allow_pickle=True)
init_npz_loose = np.load('./iter.init/iter.000000_loose.npz',allow_pickle=True)
print(len(init_npz['coord']))
coords.extend(init_npz['coord']); symbols.extend(init_npz['symbol']); conf_idxs.extend(init_npz['conf_idx'])
plqys.extend(init_npz['plqy']); e_ads.extend(init_npz['e_ad']); edmes.extend(init_npz['edme'])
homos.extend(init_npz['homo']); lumos.extend(init_npz['lumo'])


coords1.extend(init_npz_loose['coord']); symbols1.extend(init_npz_loose['symbol']); conf_idxs1.extend(init_npz_loose['conf_idx'])
plqys1.extend(init_npz_loose['plqy']); e_ads1.extend(init_npz_loose['e_ad']); edmes1.extend(init_npz_loose['edme'])
homos1.extend(init_npz_loose['homo']); lumos1.extend(init_npz_loose['lumo'])

for ii in range(13):
    init_npz = np.load('./iter.'+fmt%(ii)+'/iter.'+fmt%(ii)+'.npz',allow_pickle=True)
    init_npz_loose = np.load('./iter.'+fmt%(ii)+'/iter.'+fmt%(ii)+'.npz',allow_pickle=True)
    print(len(init_npz['coord']))
    coords.extend(init_npz['coord']); symbols.extend(init_npz['symbol']); conf_idxs.extend(init_npz['conf_idx'])
    plqys.extend(init_npz['plqy']); e_ads.extend(init_npz['e_ad']); edmes.extend(init_npz['edme'])
    homos.extend(init_npz['homo']); lumos.extend(init_npz['lumo'])

    coords1.extend(init_npz_loose['coord']); symbols1.extend(init_npz_loose['symbol']); conf_idxs1.extend(init_npz_loose['conf_idx'])
    plqys1.extend(init_npz_loose['plqy']); e_ads1.extend(init_npz_loose['e_ad']); edmes1.extend(init_npz_loose['edme'])
    homos1.extend(init_npz_loose['homo']); lumos1.extend(init_npz_loose['lumo'])

np.savez('oled_data_tight.npz',coord=coords,symbol=symbols,conf_idx=conf_idxs,plqy=plqys,e_ad=e_ads,edme=edmes,homo=homos,lumo=lumos)
np.savez('oled_data.npz',coord=coords1,symbol=symbols1,conf_idx=conf_idxs1,plqy=plqys1,e_ad=e_ads1,edme=edmes1,homo=homos1,lumo=lumos1)

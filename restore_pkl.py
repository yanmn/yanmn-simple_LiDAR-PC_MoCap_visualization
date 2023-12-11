import numpy as np 
import pickle
import h5py

f = h5py.File("D:/LiDARCap_extend/data_Climbing_v1_final/20220711_chenchen_155cm44kg13ageF/004/chenchen004_pred_restore_60_climbing.hdf5", "r")
for k in f.keys():
    print(k, type(f[k]), f[k].shape)









import numpy as np 
import h5py

f = h5py.File("D:/LiDARCap_extend/data_Climbing_v1_final/20220711_chenchen_155cm44kg13ageF/004/visual.hdf5", "r")
for k in f.keys():
    print(k, type(f[k]), f[k].shape)

point_clouds = f["point_clouds"][:,:, :, :].reshape(-1, 512, 3) 
pose = f["pose"][:, :,:].reshape(-1, 72) 
pred_rotmats = f["pred_rotmats"][:, :, :, :, :].reshape(-1, 24, 3, 3) 

print(point_clouds.shape)
print(pose.shape)
print(pred_rotmats.shape)

f=h5py.File("D:/LiDARCap_extend/data_Climbing_v1_final/20220711_chenchen_155cm44kg13ageF/004/chenchen004_pred_restore_60_climbing.hdf5","w")

f.create_dataset("point_clouds", data = point_clouds)
f.create_dataset("pred_rotmats", data = pred_rotmats)
f.create_dataset("pose", data = pose)
f.close()

print("finish")

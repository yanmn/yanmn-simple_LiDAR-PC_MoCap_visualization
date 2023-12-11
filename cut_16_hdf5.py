import numpy as np 
import h5py

f = h5py.File("D:/LiDARCap_extend/data_Climbing_v1_final/20220711_chenchen_155cm44kg13ageF/004/visual_1.hdf5", "r")
for k in f.keys():
    print(k, type(f[k]), f[k].shape)

f_o = h5py.File("D:/LiDARCap_extend/data_Climbing_v1_final/20220714_chenchen001/001/chenchen001_label_cut16pred.hdf5", "r")
for k in f_o.keys():
    print(k, type(f_o[k]), f_o[k].shape)

# full_joints <class 'h5py._hl.dataset.Dataset'> (4677, 24, 3)
# lidar_to_mocap_RT <class 'h5py._hl.dataset.Dataset'> (3, 3)
# point_clouds <class 'h5py._hl.dataset.Dataset'> (4677, 512, 3)
# points_num <class 'h5py._hl.dataset.Dataset'> (4677,)
# pose <class 'h5py._hl.dataset.Dataset'> (4677, 72)
# rotmats <class 'h5py._hl.dataset.Dataset'> (4677, 24, 3, 3)
# shape <class 'h5py._hl.dataset.Dataset'> (4677, 10)
# trans <class 'h5py._hl.dataset.Dataset'> (4677, 3)


end = f["point_clouds"].shape[0] - (f["point_clouds"].shape[0] % 16)

full_joints = f["full_joints"][:end, :, :].reshape(-1, 16, 24, 3) # (4677, 24, 3)
point_clouds = f["point_clouds"][:end, :, :].reshape(-1, 16, 512, 3) # (4677, 512, 3)
pose = f["pose"][:end, :].reshape(-1, 16, 72) # (4677, 72)
rotmats = f["rotmats"][:end, :, :, :].reshape(-1, 16, 24, 3, 3) # (4677, 24, 3, 3)
shape = f["shape"][:end, :].reshape(-1, 16, 10) # (4677, 10)
trans = f["trans"][:end, :].reshape(-1, 16, 3) # (4677, 3)

print(full_joints.shape)
print(point_clouds.shape)
print(pose.shape)
print(rotmats.shape)
print(shape.shape)
print(trans.shape)

# (292, 16, 24, 3)
# (292, 16, 512, 3)
# (292, 16, 72)
# (292, 16, 24, 3, 3)
# (292, 16, 10)
# (292, 16, 3)

# 保存 lidarcap_test_2.hdf5 chenchen001_label
f=h5py.File("/data/ym/climbing/20220714chenchen_155cm44kg13age_F/chenchen001_label_cut16.hdf5","w")

f.create_dataset("full_joints", data = full_joints)
f.create_dataset("point_clouds", data = point_clouds)
f.create_dataset("rotmats", data = rotmats)
f.create_dataset("pose", data = pose)
f.create_dataset("shape", data = shape)
f.create_dataset("trans", data = trans)
f.close()
print("chenchen001_label_2.hdf5 保存成功。")
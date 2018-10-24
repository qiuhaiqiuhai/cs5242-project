# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
import CONST
import os
import time
import plot3D
import pickle
from readpdb_example import read_pdb_test
from data_preprocess import fill_voxel, voxelise_1

if not os.path.exists(CONST.DIR.preprocess_test):
    os.makedirs(CONST.DIR.preprocess_test)

def prebind(pro_zip, lig_zip, min_dist = 1.5, max_dist = 7.5):
    min_distance_lig = list()
    for lig_atom in lig_zip:
        distance = 100
        for pro_atom in pro_zip:
            # if(pro_atom is not lig_atom):
            distance = min(distance, np.linalg.norm(np.asarray(lig_atom[:3]) - np.asarray(pro_atom[:3])))

        min_distance_lig.append(distance)
    if (np.min(min_distance_lig) > min_dist and np.max(min_distance_lig) < max_dist):
        return True

    return False

pro_id = 3
pro = read_pdb_test(pro_id, 'pro')
pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))

voxel_list = []

count = 0
for i in range(CONST.DATA.test_total):
    lig = read_pdb_test(i+1, type='lig')
    lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))
    if(not prebind(pro_zip, lig_zip)):
        continue
    voxels = []
    for lig_atom in range(len(lig['x'])):
        pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom)
        print('pro: %d, lig: %d, lig_atom: %d, atom count: %d'%(pro_id, i+1, lig_atom, neighbor_count))
        # atom_count.append(neighbor_count)
        voxels.append(voxelise_1(pre_voxel))
    voxel_list.append((i+1, voxels))
    count+=1
pickle.dump(voxel_list, open(CONST.DIR.preprocess_test+'%04d_pro.p'%pro_id, "wb"))
print(count)
# plot3D.plot_atoms(pro, lig)
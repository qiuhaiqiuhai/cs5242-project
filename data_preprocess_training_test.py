import numpy as np
import CONST
import os
import time
import plot3D
import pickle
from readpdb_example import read_pdb
from data_preprocess import fill_voxel, voxelise_1

test_dir = '../preprocessed_training_test/'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
test_indexes = np.loadtxt('testing_indexes.txt')


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

count = 0
for pro_i in test_indexes:
    pro = read_pdb(pro_i, 'pro')
    pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))
    voxel_list = []
    for lig_i in test_indexes:
        lig = read_pdb(lig_i, type='lig')
        lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))
        # check if pro and lig is within minimum distance and maximum distances
        if not prebind(pro_zip, lig_zip):
            continue
        voxels = []
        for lig_atom in range(len(lig['x'])):
            pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom)
            print('pro: %d, lig: %d, lig_atom: %d, atom count: %d'%(pro_i, lig_i, lig_atom, neighbor_count))
            voxels.append(voxelise_1(pre_voxel))
        voxel_list.append((lig_i, voxels))
    count += 1
    pickle.dump(voxel_list, open(test_dir+'%04d_pro.p'%pro_i, "wb"))
print(count)
# plot3D.plot_atoms(pro, lig)
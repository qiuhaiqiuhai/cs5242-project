import numpy as np
import CONST
import os
import time
import plot3D
import pickle
from readpdb_example import read_pdb
from data_preprocess import fill_voxel, voxelise_1, prebind


if __name__ == "__main__":
    test_dir = '../preprocessed_training_test/'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_indexes = np.loadtxt('testing_indexes.txt')

    count = 0
    for pro_i in [881, 34]:
        pro = read_pdb(pro_i, 'pro')
        voxel_list = []
        for lig_i in test_indexes:
            lig = read_pdb(lig_i, type='lig')
            # check if pro and lig is within minimum distance and maximum distances
            if not prebind(pro, lig):
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
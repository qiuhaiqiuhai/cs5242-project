import numpy as np
import CONST
import os
import time
import pickle
from readpdb_example import read_pdb
from data_preprocess import fill_voxel, voxelise_1, prebind


if __name__ == "__main__":
    size = 25
    step = 1.5
    test_dir = '../preprocessed_training_test/size%d_step%.1f/'%(size, step)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_indexes = np.loadtxt('testing_indexes.txt')

    count = 0
    for pro_i in text_indexes:
        if os.path.isfile(test_dir+'%04d_pro.p'%pro_i):
            continue
        pro = read_pdb(pro_i, 'pro')
        voxel_list = []
        for lig_i in test_indexes:
            lig = read_pdb(lig_i, type='lig')
            # check if pro and lig is within minimum distance and maximum distances
            if not prebind(pro, lig):
                continue
            voxels = []
            for lig_atom in range(len(lig['x'])):
                pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom, size=size, step=step)
                print('pro: %d, lig: %d, lig_atom: %d, atom count: %d'%(pro_i, lig_i, lig_atom, neighbor_count))
                voxels.append(voxelise_1(pre_voxel,size=size,step=step))
            voxel_list.append((lig_i, voxels))
        count += 1
        pickle.dump(voxel_list, open(test_dir+'%04d_pro.p'%pro_i, "wb"))
    print(count)
    # plot3D.plot_atoms(pro, lig)
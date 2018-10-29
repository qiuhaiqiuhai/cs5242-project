import CONST
import os
import pickle
from readpdb_example import read_pdb_test
from data_preprocess import fill_voxel, voxelise_1, prebind

if __name__ == "__main__":
    size = CONST.VOXEL.size
    step = CONST.VOXEL.step
    folder = CONST.DIR.preprocess_test%(size, step)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for pro_id in range(1, CONST.DATA.test_total+1):
        if os.path.isfile(folder+'%04d_pro.p'%pro_id):
            continue
        pro = read_pdb_test(pro_id, 'pro')
        voxel_list = []
        count = 0
        for i in range(CONST.DATA.test_total):
            lig = read_pdb_test(i+1, type='lig')
            if not prebind(pro, lig):
                continue
            voxels = []
            for lig_atom in range(len(lig['x'])):
                pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom, size=size, step=step)
                print('pro: %d, lig: %d, lig_atom: %d, atom count: %d'%(pro_id, i+1, lig_atom, neighbor_count))
                # atom_count.append(neighbor_count)
                voxels.append(voxelise_1(pre_voxel, size=size, step=step))
            voxel_list.append((i+1, voxels))
            count+=1
        pickle.dump(voxel_list, open(folder+'%04d_pro.p'%pro_id, "wb"))
    print(count)
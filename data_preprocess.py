# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
import CONST
import os
import time

def data_preprocess_bind(data_index):
    voxels = []
    pro = readpdb.read_pdb(data_index, 'pro')
    lig = readpdb.read_pdb(data_index, 'lig')

    for lig_atom in range(len(lig['x'])):
        pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom)
        print('index: %d, lig_atom: %d, atom count: %d'%(data_index, lig_atom, neighbor_count))
        # atom_count.append(neighbor_count)
        voxels.append(voxelise(pre_voxel))
        # np.save('../preprocessed_data/%04d_bind_%02d'%(data_index, lig_atom+1),voxel)
    return voxels

def data_preprocess_unbind(data_index, unbind_count = CONST.DATA.unbind_count):

    lig = readpdb.read_pdb(data_index, 'lig')
    lig_len = len(lig['x'])
    voxels = []
    count = 0
    # use only selected training set
    for i in training_indexes:
        if (i == data_index):
            continue
        pro = readpdb.read_pdb(i, 'pro')
        # trial = True
        for lig_atom in range(lig_len):

            # if(not prebind(pro, lig, min_dist=1, max_dist=10, lig_atom = lig_atom)):
            #     continue

            pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom = lig_atom)

            if(neighbor_count>0):
                count += 1
                print('index: %d, lig_atom:%d, unbind_index: %d, atom count: %d'%(data_index, lig_atom, i, neighbor_count))
                voxels.append(voxelise(pre_voxel))
                # np.save('../preprocessed_data/%04d_unbind_%02d'%(data_index, count),voxel)
                if(count == unbind_count*lig_len):
                    return voxels
            else:
                # trial = False
                # if(trial<=1):
                break
    return voxels

def prebind(pro, lig, min_dist=1.5, max_dist=7.5, lig_atom=None):
    pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))
    lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))
    if(lig_atom is not None):
        lig_zip = [lig_zip[lig_atom]]

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

def voxelise_1(pre_voxel, size = CONST.VOXEL.size, step = CONST.VOXEL.step):
    voxel = np.zeros((size, size, size, 4))

    for atom_recenter in pre_voxel:
        x1 = math.floor(atom_recenter[0] / step);
        x2 = x1 + 1
        y1 = math.floor(atom_recenter[1] / step);
        y2 = y1 + 1
        z1 = math.floor(atom_recenter[2] / step);
        z2 = z1 + 1
        channel = (atom_recenter[3] == 'h') * 2 + (atom_recenter[4] == 'r')

        bias = size // 2

        for a1, b1, c1, a2, b2, c2 in [(x1, y1, z1, x2, y2, z2),
                                       (x1, y1, z2, x2, y2, z1),
                                       (x1, y2, z1, x2, y1, z2),
                                       (x2, y1, z1, x1, y2, z2),
                                       (x2, y2, z1, x1, y1, z2),
                                       (x1, y2, z2, x2, y1, z1),
                                       (x2, y1, z2, x1, y2, z1),
                                       (x2, y2, z2, x1, y1, z1)]:

            voxel[a1 + bias, b1 + bias, c1 + bias, channel] += abs(atom_recenter[0] / step - a2) * abs(atom_recenter[1] / step - b2) * abs(atom_recenter[2] / step - c2)

    return voxel

def voxelise_2(pre_voxel, size = CONST.VOXEL.size, step = CONST.VOXEL.step):
    voxel = np.zeros((size, size, size, 4))

    for atom_recenter in pre_voxel:
        x1 = math.floor(atom_recenter[0] / step);
        x2 = x1 + 1
        y1 = math.floor(atom_recenter[1] / step);
        y2 = y1 + 1
        z1 = math.floor(atom_recenter[2] / step);
        z2 = z1 + 1
        channel = (atom_recenter[3] == 'h') * 2 + (atom_recenter[4] == 'r')

        bias = size // 2

        for a1, b1, c1, a2, b2, c2 in [(x1, y1, z1, x2, y2, z2),
                                       (x1, y1, z2, x2, y2, z1),
                                       (x1, y2, z1, x2, y1, z2),
                                       (x2, y1, z1, x1, y2, z2),
                                       (x2, y2, z1, x1, y1, z2),
                                       (x1, y2, z2, x2, y1, z1),
                                       (x2, y1, z2, x1, y2, z1),
                                       (x2, y2, z2, x1, y1, z1)]:

            voxel[a1 + bias, b1 + bias, c1 + bias, channel] = max(voxel[a1 + bias, b1 + bias, c1 + bias, channel],abs(atom_recenter[0] / step - a2) * abs(atom_recenter[1] / step - b2) * abs(atom_recenter[2] / step - c2))

    return voxel

def voxelise_3(pre_voxel, size = CONST.VOXEL.size, step = CONST.VOXEL.step):
    voxel = np.zeros((size, size, size, 4))

    for atom_recenter in pre_voxel:
        x = round(atom_recenter[0] / step);
        y = round(atom_recenter[1] / step);
        z = round(atom_recenter[2] / step);
        channel = (atom_recenter[3] == 'h') * 2 + (atom_recenter[4] == 'r')

        bias = size // 2
        voxel[x + bias, y + bias, z + bias, channel] = 1

    return voxel

def fill_voxel(pro, lig, lig_atom = 0, size = CONST.VOXEL.size, step = CONST.VOXEL.step):
    pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))
    lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))

    center = lig_zip[lig_atom]
    neighbor_count = 0

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    pre_voxel = []
    distance = 100
    for ch, cp, ca, atoms in [('m', 'g', 0.5, pro_zip), ('r', 'b', 1.0, lig_zip)]:
        # origin_x, origin_y, origin_z, origin_type = [], [], [], []

        for atom in atoms:
            if (abs(atom[0] - center[0]) < size//2 * step
                    and abs(atom[1] - center[1]) < size//2 * step
                    and abs(atom[2] - center[2]) < size//2 * step):
                neighbor_count += (ch == 'm') * 1

                atom_recenter = [atom[0] - center[0],
                                 atom[1] - center[1],
                                 atom[2] - center[2], atom[3], ch]
                pre_voxel.append(atom_recenter)

                if (ch == 'm'):
                    distance = min(distance, np.linalg.norm(np.asarray(atom_recenter[:3])))

                # origin_x.append(atom_recenter[0])
                # origin_y.append(atom_recenter[1])
                # origin_z.append(atom_recenter[2])
                # origin_type.append(atom_recenter[3])



        # h_list = [i for i, x in enumerate(origin_type) if x == 1]
        # p_list = [i for i, x in enumerate(origin_type) if x == -1]

    #     for c, m, mors in [(ch, 'o', h_list), (cp, '^', p_list)]:
    #         xs = [origin_x[i] for i in mors]
    #         ys = [origin_y[i] for i in mors]
    #         zs = [origin_z[i] for i in mors]
    #
    #         ax.scatter(xs, ys, zs, c=c, marker=m, alpha=ca)
    #
    #
    # plt.show()
    if (distance < CONST.LIMIT.min or distance > CONST.LIMIT.max):
        neighbor_count = 0

    return pre_voxel, neighbor_count

if __name__ == "__main__":
    # data_preprocess_bind(1)


    # atom_count = []
    start = time.time()
    if not os.path.exists(CONST.DIR.preprocess_base):
        os.makedirs(CONST.DIR.preprocess_base)

    training_indexes = np.loadtxt('training_indexes.txt')

    voxelises = [None, voxelise_1, voxelise_2, voxelise_3]
    '''
    for voxelise_i in [1, 2, 3]:
    
        voxelise = voxelises[voxelise_i]
    
        bind_data = []
        unbind_data = []
    
        if not os.path.exists(CONST.DIR.voxelise_base%voxelise_i):
            os.makedirs(CONST.DIR.voxelise_base%voxelise_i)
    
        for data_index in training_indexes[:CONST.DATA.processed_amount+1]:
            bind_data.extend(data_preprocess_bind(data_index))
            unbind_data.extend(data_preprocess_unbind(data_index))
    
        print("bind data: " + str(len(bind_data)))
        np.save(CONST.DIR.bind_data%voxelise_i, bind_data)
        print("unbind data: " + str(len(unbind_data)))
        data_len = 1 + len(unbind_data)//CONST.DATA.unbind_count
        for i in range(CONST.DATA.unbind_count):
            np.save(CONST.DIR.unbind_data%(voxelise_i,(i+1)), unbind_data[i*data_len:min((i+1)*data_len, len(unbind_data))])
    '''

    # use only selected training set to train
    for voxelise_i in [1]:

        voxelise = voxelises[voxelise_i]

        bind_data = []
        unbind_data = []

        if not os.path.exists(CONST.DIR.training_base%voxelise_i):
            os.makedirs(CONST.DIR.training_base%voxelise_i)

        for data_index in training_indexes[:CONST.DATA.processed_amount+1]:
            bind_data.extend(data_preprocess_bind(data_index))
            unbind_data.extend(data_preprocess_unbind(data_index))

        print("bind data: " + str(len(bind_data)))
        np.save(CONST.DIR.training_bind_data%voxelise_i, bind_data)
        print("unbind data: " + str(len(unbind_data)))
        data_len = 1 + len(unbind_data)//CONST.DATA.unbind_count
        for i in range(CONST.DATA.unbind_count):
            np.save(CONST.DIR.training_unbind_data%(voxelise_i,(i+1)), unbind_data[i*data_len:min((i+1)*data_len, len(unbind_data))])

    end = time.time()
    print(end - start)

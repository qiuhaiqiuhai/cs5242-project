# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
import CONST
import os

if not os.path.exists(CONST.DIR.preprocess_base):
    os.makedirs(CONST.DIR.preprocess_base)

def data_preprocess_bind(data_index):
    voxels = []
    pro = readpdb.read_pdb(data_index, 'pro')
    lig = readpdb.read_pdb(data_index, 'lig')

    for lig_atom in range(len(lig['x'])):
        voxel, neighbor_count = fill_voxel(pro, lig, lig_atom)
        print('index: %d, lig_atom: %d, atom count: %d'%(data_index, lig_atom, neighbor_count))
        # atom_count.append(neighbor_count)
        voxels.append(voxel)
        # np.save('../preprocessed_data/%04d_bind_%02d'%(data_index, lig_atom+1),voxel)
    return voxels

def data_preprocess_unbind(data_index, unbind_count = CONST.DATA.unbind_count):

    lig = readpdb.read_pdb(data_index, 'lig')
    lig_len = len(lig['x'])
    voxels = []
    count = 0
    for i in range(1, 3001):
        if (i == data_index):
            continue
        pro = readpdb.read_pdb(i, 'pro')
        for lig_atom in range(lig_len):

            voxel, neighbor_count = fill_voxel(pro, lig, lig_atom = lig_atom)

            if(neighbor_count>0):
                count += 1
                print('index: %d, lig_atom:%d, unbind_index: %d, atom count: %d'%(data_index, lig_atom, i, neighbor_count))
                voxels.append(voxel)
                # np.save('../preprocessed_data/%04d_unbind_%02d'%(data_index, count),voxel)
                if(count == unbind_count*lig_len):
                    return voxels
    return voxels

def fill_voxel(pro, lig, lig_atom = 0, size = CONST.VOXEL.size, step = CONST.VOXEL.step):
    pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))
    lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))

    center = lig_zip[lig_atom]
    voxel = np.zeros((size, size, size, 4))
    neighbor_count = 0

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for ch, cp, ca, atoms in [('m', 'g', 0.5, pro_zip), ('r', 'b', 1.0, lig_zip)]:
        # origin_x, origin_y, origin_z, origin_type = [], [], [], []

        for atom in atoms:
            if (abs(atom[0] - center[0]) < size//2 * step
                    and abs(atom[1] - center[1]) < size//2 * step
                    and abs(atom[2] - center[2]) < size//2 * step):
                neighbor_count += (ch == 'm') * 1
                atom_recenter = [atom[0] - center[0],
                                 atom[1] - center[1],
                                 atom[2] - center[2], atom[3]]

                # origin_x.append(atom_recenter[0])
                # origin_y.append(atom_recenter[1])
                # origin_z.append(atom_recenter[2])
                # origin_type.append(atom_recenter[3])

                x1 = math.floor(atom_recenter[0] / step); x2 = x1+1
                y1 = math.floor(atom_recenter[1] / step); y2 = y1+1
                z1 = math.floor(atom_recenter[2] / step); z2 = z1+1
                channel = (atom[3] == 1) * 2 + (ch == 'r')

                bias = size//2

                voxel[x1+bias, y1+bias, z1+bias, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z2)
                voxel[x1+bias, y1+bias, z2+bias, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z1)
                voxel[x1+bias, y2+bias, z1+bias, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z2)
                voxel[x2+bias, y1+bias, z1+bias, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z2)
                voxel[x1+bias, y2+bias, z2+bias, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z1)
                voxel[x2+bias, y2+bias, z1+bias, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z2)
                voxel[x2+bias, y1+bias, z2+bias, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z1)
                voxel[x2+bias, y2+bias, z2+bias, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z1)

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

    return voxel, neighbor_count

# data_preprocess_bind(1)


# atom_count = []

bind_data = []
unbind_data = []
for data_index in range(1, CONST.DATA.processed_amount+1):
    bind_data.extend(data_preprocess_bind(data_index))
    unbind_data.extend(data_preprocess_unbind(data_index))

print("bind data: " + str(len(bind_data)))
np.save(CONST.DIR.bind_data, bind_data)
print("unbind data: " + str(len(unbind_data)))
data_len = 1 + len(unbind_data)//CONST.DATA.unbind_count
for i in range(CONST.DATA.unbind_count):
    np.save(CONST.DIR.unbind_data%(i+1), unbind_data[i*data_len:min((i+1)*data_len, len(unbind_data))])
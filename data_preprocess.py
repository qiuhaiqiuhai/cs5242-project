from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb

data_index = 1

def data_preprocess_bind(data_index, size = 18, step = 1):
    pro = readpdb.read_pdb(data_index, 'pro')
    lig = readpdb.read_pdb(data_index, 'lig')

    voxel, neighbor_count = fill_voxel(pro, lig, size=18, step=1)
    print('index: %d, atom count: %d'%(data_index, neighbor_count))
    np.save('../preprocessed_data/%04d'%data_index,voxel)

def data_preprocess_unbind(data_index, size = 18, step = 1):
    pro = readpdb.read_pdb(data_index, 'pro')

    for i in range(data_index+1, 1000):
        lig = readpdb.read_pdb(i, 'lig')

        voxel, neighbor_count = fill_voxel(pro, lig, size = 18, step = 1)

        if(neighbor_count>0):
            print('index: %d, atom count: %d'%(data_index, neighbor_count))
            np.save('../preprocessed_data/%04d_unbind'%data_index,voxel)
            break

def fill_voxel(pro, lig, size = 18, step = 1):
    pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))
    lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))

    center = lig_zip[0]
    voxel = np.zeros((size, size, size, 4))
    neighbor_count = 0

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    for ch, cp, ca, atoms in [('m', 'g', 0.5, pro_zip), ('r', 'b', 1.0, lig_zip)]:
        # origin_x, origin_y, origin_z, origin_type = [], [], [], []

        for atom in atoms:
            if (abs(atom[0] - center[0]) < size * step / 2
                    and abs(atom[1] - center[1]) < size * step / 2
                    and abs(atom[2] - center[2]) < size * step / 2):
                neighbor_count += (ch == 'm') * 1
                atom_recenter = [atom[0] - center[0],
                                 atom[1] - center[1],
                                 atom[2] - center[2], atom[3]]

                # origin_x.append(atom_recenter[0])
                # origin_y.append(atom_recenter[1])
                # origin_z.append(atom_recenter[2])
                # origin_type.append(atom_recenter[3])

                x1, x2 = math.floor(atom_recenter[0] / step), math.ceil(atom_recenter[0] / step)
                y1, y2 = math.floor(atom_recenter[1] / step), math.ceil(atom_recenter[1] / step)
                z1, z2 = math.floor(atom_recenter[2] / step), math.ceil(atom_recenter[2] / step)
                channel = (atom[3] == 1) * 2 + (ch == 'r')

                voxel[x1, y1, z1, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y2) * abs(atom_recenter[2] / step - z2) * atom_recenter[3]
                voxel[x1, y1, z2, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z1) * atom_recenter[3]
                voxel[x1, y2, z1, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z2) * atom_recenter[3]
                voxel[x2, y1, z1, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z2) * atom_recenter[3]
                voxel[x1, y2, z2, channel] += abs(atom_recenter[0] / step - x2) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z1) * atom_recenter[3]
                voxel[x2, y2, z1, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z2) * atom_recenter[3]
                voxel[x2, y1, z2, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y2) * abs(
                    atom_recenter[2] / step - z1) * atom_recenter[3]
                voxel[x2, y2, z2, channel] += abs(atom_recenter[0] / step - x1) * abs(
                    atom_recenter[1] / step - y1) * abs(
                    atom_recenter[2] / step - z1) * atom_recenter[3]

        # h_list = [i for i, x in enumerate(origin_type) if x == 1]
        # p_list = [i for i, x in enumerate(origin_type) if x == -1]

        # for c, m, mors in [(ch, 'o', h_list), (cp, '^', p_list)]:
        #     xs = [origin_x[i] for i in mors]
        #     ys = [origin_y[i] for i in mors]
        #     zs = [origin_z[i] for i in mors]
        #
        #     ax.scatter(xs, ys, zs, c=c, marker=m, alpha=ca)

        return voxel, neighbor_count

for data_index in range(1, 101):
    data_preprocess_bind(data_index)
    data_preprocess_unbind(data_index)


# plt.show()

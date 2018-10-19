from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import CONST
import readpdb_example as readpdb


def read_pdb(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()
        # print(stripped_line)

        splitted_line = stripped_line.split('\t')

        X_list.append(float(splitted_line[0]))
        Y_list.append(float(splitted_line[1]))
        Z_list.append(float(splitted_line[2]))
        atomtype_list.append(str(splitted_line[3]))

    return {'x':X_list,
			'y':Y_list,
			'z':Z_list,
			'type':atomtype_list}


def plot_atoms(pro, lig):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for ch, cp, ca, molecule in [('m', 'g', 0.5, pro), ('r', 'b', 1.0, lig)]:
        h_list = [i for i, x in enumerate(molecule['type']) if x == "h"]
        p_list = [i for i, x in enumerate(molecule['type']) if x == "p"]


        for c, m, mors in [(ch, 'o', h_list), (cp, '^', p_list)]:
            xs = [molecule['x'][i] for i in mors]
            ys = [molecule['y'][i] for i in mors]
            zs = [molecule['z'][i] for i in mors]

            ax.scatter(xs, ys, zs, c=c, marker=m, alpha=ca)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_voxel(voxel):
    # prepare some coordinates, and attach rgb values to each
    size = CONST.VOXEL.size
    index_size = size+1
    r, g, b = np.indices((index_size, index_size, index_size))

    channel = CONST.VOXEL.ch_pro_h

    # combine the color components
    colors = np.zeros((size,size,size) + (3,))
    colors[..., 0] = voxel[:,:,:,channel]
    colors[..., 1] = 1 - voxel[:,:,:,channel]
    colors[..., 2] = 1 - voxel[:,:,:,channel]

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    voxel = voxel[:,:,:,channel]>0
    ax.voxels(r, g, b, voxel, facecolors=colors,
          edgecolors=np.clip(2*colors - 0.5, 0, 1), linewidth=0.5)
    ax.set(xlabel='r', ylabel='g', zlabel='b')

    plt.show()



# pro = read_pdb('../testing_data_release/testing_data/0009_pro_cg.pdb')
# lig = read_pdb('../testing_data_release/testing_data/0004_lig_cg.pdb')
# plot_atoms(pro, lig)

# voxel = np.load('../preprocessed_data/0001_bind_01.npy')
# plot_voxel(voxel)
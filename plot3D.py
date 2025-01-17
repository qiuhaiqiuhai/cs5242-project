from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import CONST
import readpdb_example as readpdb

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

if __name__ == "__main__":
    import sys
    if sys.argv[1] == 'train':
        pro = readpdb.read_pdb(int(sys.argv[2]), 'pro')
        lig = readpdb.read_pdb(int(sys.argv[3]), 'lig')
    elif sys.argv[1] == 'test':
        pro = readpdb.read_pdb_test(int(sys.argv[2]), 'pro')
        lig = readpdb.read_pdb_test(int(sys.argv[3]), 'lig')
    plot_atoms(pro, lig)

# voxel = np.load('../preprocessed_data/bind_data.npy')[0]
# plot_voxel(voxel)
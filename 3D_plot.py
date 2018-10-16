from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import readpdb_example as readpdb

def plot(pro, lig):
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



pro = readpdb.read_pdb(1, 'pro')
lig = readpdb.read_pdb(1, 'lig')
plot(pro, lig)
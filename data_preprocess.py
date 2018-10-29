import numpy as np
import math
import readpdb_example as readpdb
import CONST
import os
import time


def data_preprocess_bind(data_index, size=CONST.VOXEL.size, step=CONST.VOXEL.step):
    voxels = []
    pro = readpdb.read_pdb(data_index, 'pro')
    lig = readpdb.read_pdb(data_index, 'lig')

    for lig_atom in range(len(lig['x'])):
        pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom, size=size, step=step)
        print('index: %d, lig_atom: %d, atom count: %d'%(data_index, lig_atom, neighbor_count))
        voxels.append(voxelise(pre_voxel, size=size, step=step))
    return voxels


def data_preprocess_unbind(data_index, unbind_count = CONST.DATA.unbind_count, size=CONST.VOXEL.size, step=CONST.VOXEL.step):

    lig = readpdb.read_pdb(data_index, 'lig')
    lig_len = len(lig['x'])
    voxels = []
    count = 0
    # use only selected training set
    for i in training_indexes:
        if i == data_index:
            continue
        pro = readpdb.read_pdb(i, 'pro')
        # trial = True
        for lig_atom in range(lig_len):
            if not prebind(pro, lig, lig_atom = lig_atom):
                continue

            pre_voxel, neighbor_count = fill_voxel(pro, lig, lig_atom = lig_atom, size=size, step=step)

            if(neighbor_count>0):
                count += 1
                print('index: %d, lig_atom:%d, unbind_index: %d, atom count: %d'%(data_index, lig_atom, i, neighbor_count))
                voxels.append(voxelise(pre_voxel, size=size, step=step))
                if count == unbind_count*lig_len:
                    return voxels
            else:
                # trial = False
                # if(trial<=1):
                break
    return voxels


def prebind(pro, lig, min_dist=CONST.LIMIT.min, max_dist=CONST.LIMIT.max, lig_atom=None):
    """

    :param pro: protein
    :param lig: ligand
    :param min_dist: minimum distance limit between two atoms in protein and ligand
    :param max_dist: maximum distance limit between two atoms in protein and ligand
    :param lig_atom:
    :return:
    """
    pro_zip = list(zip(pro['x'], pro['y'], pro['z'], pro['type']))
    lig_zip = list(zip(lig['x'], lig['y'], lig['z'], lig['type']))
    if lig_atom is not None:
        lig_zip = [lig_zip[lig_atom]]

    min_distance_lig = list()
    for lig_atom in lig_zip:
        distance = 100
        for pro_atom in pro_zip:
            # if(pro_atom is not lig_atom):
            distance = min(distance, np.linalg.norm(np.asarray(lig_atom[:3]) - np.asarray(pro_atom[:3])))

        min_distance_lig.append(distance)
    if np.min(min_distance_lig) > min_dist and np.max(min_distance_lig) < max_dist:
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

    pre_voxel = []
    distance = 100
    for ch, cp, ca, atoms in [('m', 'g', 0.5, pro_zip), ('r', 'b', 1.0, lig_zip)]:

        for atom in atoms:
            if (abs(atom[0] - center[0]) < size//2 * step
                    and abs(atom[1] - center[1]) < size//2 * step
                    and abs(atom[2] - center[2]) < size//2 * step):
                neighbor_count += (ch == 'm') * 1

                atom_recenter = [atom[0] - center[0],
                                 atom[1] - center[1],
                                 atom[2] - center[2], atom[3], ch]
                pre_voxel.append(atom_recenter)

                if ch == 'm':
                    distance = min(distance, np.linalg.norm(np.asarray(atom_recenter[:3])))

    if distance < CONST.LIMIT.min or distance > CONST.LIMIT.max:
        neighbor_count = 0

    return pre_voxel, neighbor_count


if __name__ == "__main__":

    start = time.time()

    training_indexes = range(1, CONST.DATA.processed_amount+1)
    size = CONST.VOXEL.size
    step = CONST.VOXEL.step
    unbind_count = CONST.DATA.unbind_count

    training_dir = CONST.DIR.preprocess_data%(size,step)
    print (training_dir)
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    bind_dir = training_dir+'bind_data'
    unbind_dir = training_dir+'unbind_data_%02d'

    voxelise = voxelise_1

    bind_data = []
    unbind_data = []

    for data_index in training_indexes:
        bind_data.extend(data_preprocess_bind(data_index, size=size, step=step))
        unbind_data.extend(data_preprocess_unbind(data_index, unbind_count=unbind_count, size=size, step=step))

    print("bind data: " + str(len(bind_data)))
    np.save(bind_dir, bind_data)
    print("unbind data: " + str(len(unbind_data)))
    data_len = 1 + len(unbind_data)//CONST.DATA.unbind_count
    for i in range(CONST.DATA.unbind_count):
        np.save(unbind_dir%(i+1), unbind_data[i*data_len:min((i+1)*data_len, len(unbind_data))])

    end = time.time()
    print(end - start)

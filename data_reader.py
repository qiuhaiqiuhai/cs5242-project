# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
from keras.utils import to_categorical
import CONST
import plot3D


file_dir_bind = "../preprocessed_data/%04d_bind_%02d.npy"
file_dir_unbind = "../preprocessed_data/%04d_unbind_%02d.npy"

def read_processed_data(bind_count, unbind_count):
    train_x = [] ; train_y = []
    class_name = ['bind', 'unbind']

    bind_data = np.load(CONST.DIR.bind_data + '.npy')
    print(class_name[0] + ' total:' + str(bind_data.shape[0]) + ' take:' + str(bind_count))
    bind_data = bind_data[:bind_count]

    unbind_data = np.load(CONST.DIR.unbind_data + '.npy')
    print(class_name[1] + ' total:' + str(unbind_data.shape[0]) + ' take:' + str(unbind_count))
    unbind_data = unbind_data[:unbind_count]
    # for cat_num, file_dir, max in [(0, file_dir_bind, CONST.DATA.lig_data_max), (1, file_dir_unbind, unbind_count)]:
    #     count = 0
    #     for i in range(0, CONST.DATA.processed_amount):
    #         for j in range(0, max):
    #             try:
    #                 train_x.append(np.load(file_dir%(i+1, j+1)))
    #                 train_y.append(cat_num)
    #                 count+=1
    #             except FileNotFoundError:
    #                 break
    #     print(class_name[cat_num] + ':' + str(count))

    return np.append(bind_data, unbind_data, axis=0), to_categorical(np.append(np.zeros(bind_count), np.ones(unbind_count),axis=0)), class_name

# read_processed_data(5, 10)
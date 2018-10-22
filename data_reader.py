# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
from keras.utils import to_categorical
import CONST
from sklearn.utils import shuffle
import plot3D


file_dir_bind = "../preprocessed_data/%04d_bind_%02d.npy"
file_dir_unbind = "../preprocessed_data/%04d_unbind_%02d.npy"

def read_processed_data(bind_count = None, unbind_count = None, shuffled = True):
    train_x = [] ; train_y = []
    class_name = ['bind', 'unbind']

    bind_data_origin = np.load(CONST.DIR.bind_data + '.npy', mmap_mode='r')

    if(shuffled):
        bind_data_origin = shuffle(bind_data_origin)
    bind_data = bind_data_origin[:bind_count]
    print(class_name[0] + ' total:' + str(bind_data_origin.shape[0]) + ' take:' + str(len(bind_data)))

    unbind_data_origin = None
    for i in range(CONST.DATA.unbind_count):
        unbind_data_sub = np.load(CONST.DIR.unbind_data% (i + 1) + '.npy')
        print(CONST.DIR.unbind_data% (i + 1) + ' total:' + str(unbind_data_sub.shape[0]) )
        if(unbind_data_origin is None):
            unbind_data_origin = unbind_data_sub
        else:
            unbind_data_origin = np.append(unbind_data_origin, unbind_data_sub, axis=0)

        if(unbind_count and unbind_data_origin.shape[0]>=unbind_count):
            break;

    if(shuffled):
        unbind_data_origin = shuffle(unbind_data_origin)
    unbind_data = unbind_data_origin[:unbind_count]
    print(class_name[0] + ' total:' + str(unbind_data_origin.shape[0]) + ' take:' + str(len(unbind_data)))


    return np.append(bind_data, unbind_data, axis=0), to_categorical(np.append(np.zeros(len(bind_data)), np.ones(len(unbind_data)),axis=0)), class_name

# read_processed_data()
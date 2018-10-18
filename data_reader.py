from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import readpdb_example as readpdb
from keras.utils import to_categorical

path = "../preprocessed_data/%d_unbind/"
file_dir_bind = "%04d.npy"
file_dir_unbind = "%04d_unbind"

def read_processed_data(min_index, max_index, n_unbind):
    train_x = [] ; train_y = []
    test_x = [] ; test_y = []
    dir = path%n_unbind

    for i in range(min_index, max_index):
        train_x.append(np.load(os.path.join(dir, file_dir_bind%i)))
        train_y.append(0)

    for i in range(min_index, max_index):
        for file in os.listdir(dir):
            if os.path.isfile(os.path.join(dir,file)) and (file_dir_unbind%i) in file:
                train_x.append(np.load(os.path.join(dir,file)))
                train_y.append(1)

    class_name = ['bind', 'unbind']

    return np.stack(train_x, 0), to_categorical(np.array(train_y)), class_name


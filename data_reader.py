from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
from keras.utils import to_categorical

file_dir_bind = "../preprocessed_data/%04d.npy"
file_dir_unbind = "../preprocessed_data/%04d_unbind.npy"

def read_processed_data():
    train_x = [] ; train_y = []
    test_x = [] ; test_y = []

    for i in range(1, 101):
        train_x.append(np.load(file_dir_bind%i))
        train_y.append(0)

    for i in range(1, 101):
        train_x.append(np.load(file_dir_unbind%i))
        train_y.append(1)

    class_name = ['bind', 'unbind']

    return np.stack(train_x, 0), to_categorical(np.array(train_y)), np.stack(train_x, 0), to_categorical(np.array(train_y)), class_name


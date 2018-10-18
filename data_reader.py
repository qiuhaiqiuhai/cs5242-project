from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import math
import readpdb_example as readpdb
from keras.utils import to_categorical
import CONST

file_dir_bind = "../preprocessed_data/%04d_bind_%02d.npy"
file_dir_unbind = "../preprocessed_data/%04d_unbind_%02d.npy"

def read_processed_data():
    train_x = [] ; train_y = []
    test_x = [] ; test_y = []
    class_name = ['bind', 'unbind']

    for cat_num, file_dir, max in [(0, file_dir_bind, CONST.DATA.lig_data_max), (1, file_dir_unbind, CONST.DATA.unbind_count)]:
        count = 0
        for i in range(0, CONST.DATA.processed_amount):
            for j in range(0, max):
                try:
                    train_x.append(np.load(file_dir%(i+1, j+1)))
                    train_y.append(cat_num)
                    count+=1
                except FileNotFoundError:
                    break
        print(class_name[cat_num] + ':' + str(count))



    return np.stack(train_x, 0), to_categorical(np.array(train_y)), class_name

# read_processed_data()
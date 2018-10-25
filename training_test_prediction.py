from data_reader import read_processed_training_test
from keras.models import model_from_json
import numpy as np
import CONST
import pickle
import os
import plot3D

file_name = 'selected_models/box_size=19,step=1,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,' \
            'train_ratio=0.8,retrain=0_07-loss=0.10-acc=0.97'
test_indexes = np.loadtxt('testing_indexes.txt')

n = 10

def build_model():
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(file_name+'.h5')
    return model


def predict(predicts):
    print ("mean prediction")
    predicts.sort(key=lambda x: np.mean(x[1][:, 0]), reverse=True)
    lig_label = []
    lig_mean_indexes = []
    lig_min_indexes = []
    limit = min(n, len(predicts))
    for i in range(limit):
        label = 'lig_%04d' % predicts[i][0]
        # print('lig_%04d' % predicts[i][0], predicts[i][1][:, 0])
        lig_label.append('lig_%04d' % predicts[i][0])
        lig_mean_indexes.append(predicts[i][0])
        # print('lig_%d' % predicts[i][0], end=' ')

    # max
    # print()
    # print('*' * 30)
    # print("min prediction")
    predicts.sort(key=lambda x: np.min(x[1][:, 0]), reverse=True)
    for i in range(limit):
        # print('lig_%04d' % predicts[i][0], predicts[i][1][:, 0])
        lig_label.append('lig_%04d' % predicts[i][0])
        lig_min_indexes.append(predicts[i][0])

    # append dummy index to make recommendation count always equal
    for i in range(n-limit):
        lig_min_indexes.append(0)
        lig_mean_indexes.append(0)

    # print(np.reshape(lig_label, (2, limit)).transpose())
    return lig_mean_indexes, lig_min_indexes

mean_results = []
min_results = []
mean_count = 0
min_count = 0
model = build_model()
for pro_i in test_indexes:
    print ("******** predicting on index {0} ********".format(pro_i))
    voxels_list = read_processed_training_test(pro_id=pro_i)
    predicts = []
    for i in range(len(voxels_list)):
        # predict voxels of lig i
        predict_y = model.predict(np.asarray(voxels_list[i][1]))
        # append (ligand_index, prediction_list)
        predicts.append((voxels_list[i][0],predict_y))
    lig_mean_indexes, lig_min_indexes = predict(predicts)
    print('*** mean ***' )
    print('pro_%04d'%pro_i, lig_mean_indexes)
    if pro_i in lig_mean_indexes:
        print('Got it!')
        mean_count += 1;
    labels = ['pro_%04d'%pro_i]
    mean_results.append(labels.extend(['lig_%04d'%index for index in lig_mean_indexes]))
    print('*** min ***' )
    print('pro_%04d' % pro_i, lig_min_indexes)
    if pro_i in lig_min_indexes:
        print('Got it!')
        min_count += 1;
    labels = ['pro_%04d'%pro_i]
    min_results.append(labels.extend(['lig_%04d'%index for index in lig_min_indexes]))


np.savetxt('training_test_set_result_using_mean.txt', mean_results)
np.savetxt('training_test_set_result_using_min.txt', min_results)
print ('using mean: conversion rate for top 10 predictions: {0}'.format(mean_count/len(test_indexes)))
print ('using min: conversion rate for top 10 predictions: {0}'.format(min_count/len(test_indexes)))


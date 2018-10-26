from data_reader import read_processed_training_test
from keras.models import model_from_json
import numpy as np
import CONST
import datetime
import pickle
import os
import plot3D


test_indexes = np.loadtxt('testing_indexes.txt')

n = 10


if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def build_model(file_name):
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(file_name+'.h5')
    return model


def predict(predicts):
    lig_mean_indexes = []
    lig_min_indexes = []
    lig_mean_square_indexes = []
    limit = min(n, len(predicts))
    print_limit = min(20, len(predicts))

    print ("mean prediction")
    predicts.sort(key=lambda x: np.mean(x[1][:, 0]), reverse=True)
    for i in range(limit):
        lig_mean_indexes.append(predicts[i][0])
    for i in range(print_limit):
        label = 'lig_%04d' % predicts[i][0]
        print(label, predicts[i][1][:, 0])

    print ('*' * 30)
    print("min prediction")
    predicts.sort(key=lambda x: np.min(x[1][:, 0]), reverse=True)
    for i in range(limit):
        lig_min_indexes.append(predicts[i][0])
    for i in range(print_limit):
        label = 'lig_%04d' % predicts[i][0]
        print(label, predicts[i][1][:, 0])

    print('*' * 30)
    print("mean square prediction")
    predicts.sort(key=lambda x: np.mean(x[1][:, 0]**2), reverse=True)
    for i in range(limit):
        lig_mean_square_indexes.append(predicts[i][0])
    for i in range(print_limit):
        label = 'lig_%04d' % predicts[i][0]
        print(label, predicts[i][1][:, 0])

    # append dummy index to make recommendation count always equal
    for i in range(n-limit):
        lig_min_indexes.append(0)
        lig_mean_indexes.append(0)
        lig_mean_square_indexes.append(0)

    # print(np.reshape(lig_label, (2, limit)).transpose())
    return lig_mean_indexes, lig_min_indexes, lig_mean_square_indexes

def save_prediction_result(indexes, model_filename, test_data_dir, result_dir):
    model = build_model(model_filename)
    for pro_i in indexes:
        print ("saving result for {0}".format(pro_i))
        voxels_list = read_processed_training_test(pro_id=pro_i, directory=test_data_dir)
        predicts = []
        for i in range(len(voxels_list)):
            # predict voxels of lig i
            predict_y = model.predict(np.asarray(voxels_list[i][1]))
            # append (ligand_index, prediction_list)
            predicts.append((voxels_list[i][0],predict_y))
        pickle.dump(predicts, open(result_dir+"result_%04d.p" % pro_i, "wb"))

def predict_results(result_dir):

    mean_results = []
    min_results = []
    mean_square_results = []
    mean_count = 0
    min_count = 0
    mean_square_count = 0
    for pro_i in test_indexes:
        print("********************* predicting on index {0} ***********************".format(pro_i))
        predicts = pickle.load(open(result_dir+"result_%04d.p" % pro_i, "rb"))
        print('total ligand counts: {0}'.format(len(predicts)))
        lig_mean_indexes, lig_min_indexes, lig_mean_square_indexes = predict(predicts)
        # print('*** mean ***' )
        if pro_i in lig_mean_indexes:
            #print('Got it!')
            mean_count += 1;
        else:
            print('*** mean ***')
            print ('miss it!')
            print('pro_%04d' % pro_i, lig_mean_indexes)
        labels = ['pro_%04d'%pro_i]
        mean_results.append(np.append(labels, ['lig_%04d' % index for index in lig_mean_indexes]))
        print (mean_results)
        #print('*** min ***' )
        #print('pro_%04d' % pro_i, lig_min_indexes)
        if pro_i in lig_min_indexes:
            #print('Got it!')
            min_count += 1;
        else:
            print('*** min ***')
            print('miss it!')
            print('pro_%04d' % pro_i, lig_min_indexes)
        labels = ['pro_%04d'%pro_i]
        min_results.append(np.append(labels,['lig_%04d'%index for index in lig_min_indexes]))


        if pro_i in lig_mean_square_indexes:
            #print('Got it!')
            mean_square_count += 1;
        else:
            print('*** mean square ***')
            print('miss it!')
            print('pro_%04d' % pro_i, lig_min_indexes)
        labels = ['pro_%04d'%pro_i]
        mean_square_results.append(np.append(labels,['lig_%04d'%index for index in lig_mean_square_indexes]))

    print ('using mean: conversion rate for top 10 predictions: {0}'.format(mean_count/len(test_indexes)))
    print ('using min: conversion rate for top 10 predictions: {0}'.format(min_count/len(test_indexes)))
    print ('using mean square: conversion rate for top 10 predictions: {0}'.format(mean_square_count/len(test_indexes)))

    np.savetxt(result_dir+'training_test_result_using_mean.txt', mean_results, fmt="%s")
    np.savetxt(result_dir+'training_test_result_using_min.txt', min_results, fmt='%s')
    np.savetxt(result_dir+'training_test_result_using_mean_square.txt', mean_square_results, fmt='%s')


if __name__ == '__main__':
    date = datetime.datetime
    folder_name = date.today().strftime('%Y-%m-%d_%H_%M_%S')
    result_dir = '../training_test_result/' + folder_name + '/'

    size = 25
    step = 1.5
    test_data_dir = '../preprocessed_training_test/size%d_step%.1f/' % (size, step)
    model_file_name = 'selected_models/box_size=19,step=1,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_07-loss=0.10-acc=0.97'

    save_prediction_result(test_indexes, model_file_name, test_data_dir, result_dir)
    predict_results(result_dir)
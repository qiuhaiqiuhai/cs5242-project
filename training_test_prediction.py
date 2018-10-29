from data_reader import read_processed_training_test
from keras.models import model_from_json
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import random
import pandas as pd




n = 10

print_n = 0
test_indexes = np.sort(np.loadtxt('testing_indexes.txt'))
test_indexes = [int(i) for i in test_indexes]

def build_model(file_name):
    json_file = open(file_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(file_name+'.h5')
    return model


def predict(predicts):
    predicts.sort(key=lambda x: np.mean(x[1][:, 0]), reverse=True)
    lig_mean_indexes = [predicts[i][0] for i in range(len(predicts))]

    predicts.sort(key=lambda x: np.min(x[1][:, 0]), reverse=True)
    lig_min_indexes = [predicts[i][0] for i in range(len(predicts))]

    predicts.sort(key=lambda x: np.mean(x[1][:, 0]**2), reverse=True)
    lig_mean_square_indexes = [predicts[i][0] for i in range(len(predicts))]
    return lig_mean_indexes, lig_min_indexes, lig_mean_square_indexes


def save_prediction_result(indexes, model_filename, test_data_dir, result_dir):
    print ('save prediction result for %s'%model_filename)
    model = build_model(model_filename)
    for pro_i in indexes:
        if os.path.isfile(result_dir+'result_%04d.p'%pro_i):
            print ("skip saving result for {0} since it exists".format(pro_i))
            continue
        print ("saving result for {0}".format(pro_i))
        voxels_list = read_processed_training_test(pro_id=pro_i, directory=test_data_dir)
        predicts = []
        for i in range(len(voxels_list)):
            # predict voxels of lig i
            predict_y = model.predict(np.asarray(voxels_list[i][1]))
            # append (ligand_index, prediction_list)
            predicts.append((voxels_list[i][0],predict_y))
        pickle.dump(predicts, open(result_dir+"result_%04d.p" % pro_i, "wb"))

def predict_results(indexes, result_dir):

    mean_results = []
    min_results = []
    mean_square_results = []
    mean_count = 0
    min_count = 0
    mean_square_count = 0

    for pro_i in indexes:
        #print("********************* predicting on index {0} ***********************".format(pro_i))
        predicts = pickle.load(open(result_dir+"result_%04d.p" % pro_i, "rb"))
        #print('total ligand counts: {0}'.format(len(predicts)))
        lig_mean_indexes, lig_min_indexes, lig_mean_square_indexes = predict(predicts)
        #print('*** mean ***' )
        # if lig_mean_indexes.index(pro_i) < 10:
        #     #print('Got it!')
        #     mean_count += 1;
        # else:
        #     print('*** mean ***')
        #     print ('miss it!')
        #     print('pro_%04d' % pro_i, lig_mean_indexes)
        mean_results.append(np.append([pro_i], lig_mean_indexes))
        #print('*** min ***' )
        #print('pro_%04d' % pro_i, lig_min_indexes)
        # if lig_min_indexes.index(pro_i) < 10:
        #     #print('Got it!')
        #     min_count += 1;
        # else:
        #     print('*** min ***')
        #     print('miss it!')
        #     print('pro_%04d' % pro_i, lig_min_indexes)
        min_results.append(np.append([pro_i],lig_min_indexes))


        # if lig_mean_square_indexes.index(pro_i) < 10:
        #         #     #print('Got it!')
        #         #     mean_square_count += 1;
        # else:
            # print('*** mean square ***')
            # print('miss it!')
            # print('pro_%04d' % pro_i, lig_min_indexes)
        mean_square_results.append(np.append([pro_i],lig_mean_square_indexes))


    # print ('using mean: conversion rate for top 10 predictions: {0}'.format(mean_count/len(indexes)))
    # print ('using min: conversion rate for top 10 predictions: {0}'.format(min_count/len(indexes)))
    # print ('using mean square: conversion rate for top 10 predictions: {0}'.format(mean_square_count/len(indexes)))

    # np.savetxt(result_dir+'training_test_result_using_mean.txt', mean_results, fmt="%s")
    # np.savetxt(result_dir+'training_test_result_using_min.txt', min_results, fmt='%s')
    # np.savetxt(result_dir+'training_test_result_using_mean_square.txt', mean_square_results, fmt='%s')

    return mean_results, min_results, mean_square_results

def calculate_hist(results):

    results = np.asarray(results)
    counts = []
    for result in results:
        rank = list(result[1:]).index(result[0])+1
        counts.append(rank)
    return counts

def calculate_counts(results):
    results = np.asarray(results)
    ranks = [list(result[1:]).index(result[0]) + 1 for result in results if result[0] in list(result[1:])]
    counts = np.zeros(max(ranks)+1)
    for rank in ranks:
        counts[rank:] += 1
    return counts[1:11]/600

def training_test_prediction():
    prediction_dir = 'prediction_result/'
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    model_19_1_unbind1 = 'box_size=19,step=1,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_07-loss=0.10-acc=0.97'
    model_19_1_unbind2 = 'box_size=19,step=1.0,epochs=10,unbind=2,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_06-loss=0.09-acc=0.97'
    model_25_15_unbind1 = 'box_size=25,step=1.5,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_06-loss=0.08-acc=0.97'
    model_25_15_unbind2 = 'box_size=25,step=1.5,epochs=10,unbind=2,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_05-loss=0.07-acc=0.98'
    model_25_15_unbind3 = 'box_size=25,step=1.5,epochs=10,unbind=3,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_07-loss=0.07-acc=0.98'
    model_19_15_unbind1 = 'box_size=19,step=1.5,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_08-loss=0.08-acc=0.98'
    model_19_15_unbind2 = 'box_size=19,step=1.5,epochs=10,unbind=2,model=test4,voxelise=1,repeat=0,' \
                'train_ratio=0.8,retrain=0_09-loss=0.06-acc=0.98'

    model_file_names = [model_19_1_unbind1, model_19_1_unbind2, model_25_15_unbind1, model_25_15_unbind2,model_25_15_unbind3,
                        model_19_15_unbind1, model_19_15_unbind2]
    result_file_names = ['model_19_1_unbind1', 'model_19_1_unbind2', 'model_25_15_unbind1', 'model_25_15_unbind2','model_25_15_unbind3',
                         'model_19_15_unbind1', 'model_19_15_unbind2']
    selected=[1, 6, 3]
    model_file_names = [model_file_names[i] for i in selected]
    result_file_names = [result_file_names[i] for i in selected]

    ################ hist gram ###################
    fig, axs = plt.subplots(len(model_file_names), 1, tight_layout=True)
    for i in range(len(model_file_names)):
        model_file_name = model_file_names[i]
        result_file_name = result_file_names[i]
        result_dir = '../training_test_result/' + model_file_name + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        print(model_file_name)
        # save_prediction_result(test_indexes, 'selected_models/'+model_file_name, test_data_dir, result_dir)
        mean_results, min_results, mean_square_results = predict_results(test_indexes, result_dir)

        axs[i].hist(calculate_hist(mean_results), density=1, histtype='bar')
        #axs[i].set_xticks(range(1, 21))
        axs[i].set_title(result_file_name)
    fig.show()

    ############## bar chart ################
    width = 0.2
    fig, ax = plt.subplots(1, 1)
    for i in range(len(model_file_names)):
        model_file_name = model_file_names[i]
        result_file_name = result_file_names[i]
        result_dir = '../training_test_result/' + model_file_name + '/'
        mean_results, min_results, mean_square_results = predict_results(test_indexes, result_dir)

        x = np.arange(1, 11)
        ax.set_title('Prediction counts on test set, '+result_file_name)
        ax.bar(x-0.2, calculate_counts(mean_results),label='mean', width=0.2)
        ax.bar(x, calculate_counts(min_results), label='min', width=0.2)
        ax.bar(x+0.2, calculate_counts(mean_square_results), label='mean square', width=0.2)
        ax.set_xlabel('top x predictions')
        ax.set_ylabel('correction prediction counts')
        ax.legend()
    fig.show()

    ############## bar chart compare different models ###############
    fig, ax = plt.subplots(1, 1)
    x = np.arange(1, 11)
    ax.set_title('Compare of three input shape')
    for i in range(len(model_file_names)):
        result_file_name = result_file_names[i]
        model_file_name = model_file_names[i]
        result_dir = '../training_test_result/' + model_file_name + '/'
        mean_results, min_results, mean_square_results = predict_results(test_indexes, result_dir)
        print (calculate_counts(mean_results))
        ax.bar(x + width*(i-len(model_file_names)/2), calculate_counts(mean_results), label=result_file_name, width=width)

        ax.set_xlabel('top x predictions')
        ax.set_ylabel('correction prediction counts')
        ax.legend()
    fig.show()


if __name__ == '__main__':
    #training_test_prediction()
    final_model = 'box_size=19,step=1.5,epochs=10,unbind=2.0,model=test4,voxelise=1,repeat=1,retrain=0_07-loss=0.0630-acc=0.9780'
    result_dir = '../training_test_result/' + final_model + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save_prediction_result(test_indexes, 'selected_models/'+final_model, test_data_dir, result_dir)
    mean_results, min_results, mean_square_results = predict_results(range(1, 825), result_dir)


    first_line = ['pro_id']
    for i in range(1, 11):
        first_line.append('lig%d_id'%i)
    prediction_txt = [np.asarray(first_line)]
    for result in mean_results:
        line = [random.randint(1, 825) for i in range(11)]
        for i in range(min(11, len(result))):
            line[i] = '%d'%result[i]
        prediction_txt.append(line)
    np.savetxt('test_predictions.txt', prediction_txt, fmt='%s', delimiter='\t')

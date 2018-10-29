from data_reader import read_processed_test
from keras.models import model_from_json
import numpy as np
import pickle
import os
import random
import CONST


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
        voxels_list = read_processed_test(pro_id=pro_i, directory=test_data_dir)
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

    for pro_i in indexes:
        predicts = pickle.load(open(result_dir+"result_%04d.p" % pro_i, "rb"))
        lig_mean_indexes, lig_min_indexes, lig_mean_square_indexes = predict(predicts)
        mean_results.append(np.append([pro_i], lig_mean_indexes))
        min_results.append(np.append([pro_i],lig_min_indexes))
        mean_square_results.append(np.append([pro_i],lig_mean_square_indexes))

    return mean_results, min_results, mean_square_results


if __name__ == '__main__':
    n = 10
    size = CONST.VOXEL.size
    step = CONST.VOXEL.step
    final_model = CONST.MODEL.name
    result_dir = CONST.DIR.test_result + final_model + '/'
    test_data_dir = CONST.DIR.preprocess_test%(size, step)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # save prediction results in result_dir
    save_prediction_result(range(1, CONST.DATA.test_total+1), 'selected_models/'+final_model, test_data_dir, result_dir)
    # load prection results, summarize and sort based on all atoms in ligand
    mean_results, min_results, mean_square_results = predict_results(range(1, CONST.DATA.test_total+1), result_dir)

    first_line = ['pro_id']
    for i in range(1, n+1):
        first_line.append('lig%d_id'%i)
    prediction_txt = [np.asarray(first_line)]
    for result in mean_results:
        line = [random.randint(1, CONST.DATA.test_total+1) for i in range(n+1)]
        for i in range(min(n+1, len(result))):
            line[i] = '%d'%result[i]
        prediction_txt.append(line)
    np.savetxt('test_predictions.txt', prediction_txt, fmt='%s', delimiter='\t')

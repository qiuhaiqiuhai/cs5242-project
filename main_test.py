from models import test4_3Dcnn as test_network
from keras import optimizers, losses
from data_reader import read_processed_test, read_processed_data
from keras.models import model_from_json, load_model
import numpy as np
import CONST
import pickle
import os
import plot3D

def build_model():
    with open('selected_models/box_size=19,step=1,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,retrain=0_0.97.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('selected_models/box_size=19,step=1,epochs=10,unbind=1,model=test4,voxelise=1,repeat=0,retrain=0_0.97.h5')
    print(model.summary())
    return model
# train_x, trian_y, class_name = read_processed_data(10,10)
# predict_y = model.predict(np.asarray(train_x))
# print(predict_y)

def calc_res(pro_id):
    voxels_list = read_processed_test(pro_id=pro_id)
    plot3D.plot_voxel((voxels_list[0][1][0]))
    model = build_model()
    predicts = []
    for i in range(len(voxels_list)):
        predict_y = model.predict(np.asarray(voxels_list[i][1]))
        predicts.append((voxels_list[i][0],predict_y))

    pickle.dump(predicts, open("../test_result/result_%04d.p"%pro_id, "wb"))
    return predicts

if __name__ == "__main__":
    import sys
    if not os.path.exists('../test_result/'):
        os.makedirs('../test_result/')
    np.set_printoptions(precision=3, suppress=True)
    pro_id = int(sys.argv[1])
    # predicts = calc_res(pro_id)
    predicts = pickle.load(open("../test_result/result_%04d.p"%pro_id, "rb"))
    # mean
    predicts.sort(key=lambda x: np.mean(x[1][:,0]), reverse=True)
    lig_label = []
    limit = 10
    for i in range(limit):
        print('lig_%03d'%predicts[i][0], predicts[i][1][:,0])
        lig_label.append('lig_%03d'%predicts[i][0])
        # print('lig_%d' % predicts[i][0], end=' ')
    # # max
    # # print()
    # print('*'*30)
    # predicts.sort(key=lambda x: np.min(x[1][:,0]), reverse=True)
    # for i in range(limit):
    #     print('lig_%03d'%predicts[i][0], predicts[i][1][:,0])
    #     lig_label.append('lig_%03d' % predicts[i][0])

    # mean square
    print('*'*30)
    predicts.sort(key=lambda x: np.mean(x[1][:,0]**2), reverse=True)
    for i in range(limit):
        print('lig_%03d'%predicts[i][0], predicts[i][1][:,0])
        lig_label.append('lig_%03d' % predicts[i][0])

    print(np.reshape(lig_label,(2, limit)).transpose())


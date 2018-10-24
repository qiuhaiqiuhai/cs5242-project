from models import trial_3Dcnn, test1_3Dcnn, test2_3Dcnn, test3_3Dcnn, test4_3Dcnn
from keras import optimizers, losses
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model, model_from_json
from data_reader import read_processed_data
from sklearn.utils import shuffle
import logging, math
import sys, os, datetime
import numpy as np
import CONST
import glob
from shutil import copyfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger('main')
date = datetime.datetime
folder_name = date.today().strftime('%Y-%m-%d_%H_%M_%S')

dir = 'results/%s/'%folder_name
selected_dir = 'selected_models/'
if not os.path.exists(dir):
    os.makedirs(dir)
if not os.path.exists(selected_dir):
    os.makedirs(selected_dir)
model_dir = 'models/%s/'%folder_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

size = CONST.VOXEL.size
step = CONST.VOXEL.step
epochs = 10
input_shape = (size, size, size, 4)
processed_amount = CONST.DATA.processed_amount
n_bind = 10000
n_retrain = 0 # retrain how many times. If no need to retrain, put 0
selected_acc = 0.96
n_repeat = 1


# define models
model_names = ['test0', 'test1', 'test2', 'test3', 'test4']
models = []
models.append(trial_3Dcnn(input_shape=input_shape))
models.append(test1_3Dcnn(input_shape=input_shape))
models.append(test2_3Dcnn(input_shape=input_shape))
models.append(test3_3Dcnn(input_shape=input_shape))
models.append(test4_3Dcnn(input_shape=input_shape))
optimizer = optimizers.adadelta()
earlystopper = EarlyStopping(patience=2, verbose=2, monitor='val_loss')

def get_model(index):
    if index == 0:
        return trial_3Dcnn(input_shape=input_shape)
    if index == 1:
        return test1_3Dcnn(input_shape=input_shape)
    if index == 2:
        return test2_3Dcnn(input_shape=input_shape)
    if index == 3:
        return test3_3Dcnn(input_shape=input_shape)
    if index == 4:
        return test4_3Dcnn(input_shape=input_shape)

def save_model_info(file_name, model, h):
    logger.info('save model weights in {0}...'.format(model_dir+file_name))
    #model.save_weights(os.path.join(model_dir, file_name+'.h5'))
    model_json = model.to_json()
    with open(os.path.join(model_dir, file_name+'.json'), "w") as json_file:
        json_file.write(model_json)

    logger.info('save results in {0}...'.format(dir+file_name))
    np.savetxt(os.path.join(dir, file_name+ '.txt'), \
               np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))

    best_acc = max(h.history['val_acc'])
    if best_acc >= selected_acc:
        logger.info('model acc >= {1}, is saved as {0}...'.format(selected_dir+file_name, selected_acc))
        # model.save_weights(os.path.join(selected_dir, file_name + '_%.2f.h5'%best_acc))
        # look for the model weights with best acc and copy it
        files = glob.glob(model_dir+file_name+'*.h5')
        files.sort()
        best_model_dir = files[-1]
        selected_model_dir = best_model_dir.replace(model_dir, selected_dir)
        selected_json_dir = selected_model_dir.replace('.h5', '.json')
        copyfile(best_model_dir, selected_model_dir)
        with open(os.path.join(selected_json_dir), "w") as json_file:
            json_file.write(model_json)

def load_model(file_name):
    logger.info('load model in {0}...'.format(model_dir + file_name))
    with open(os.path.join(model_dir, file_name + '.json'), 'r') as f:
        json = f.read()
    loaded_model = model_from_json(json)
    loaded_model.load_weights(os.path.join(model_dir, file_name + '.h5'))
    return loaded_model

def split_data(x, y, split=0.2):
    n_train = math.floor(len(x)*(1-split))
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]

for voxelise_i in [1]:
    logger.info("using voxelise_{0}".format(voxelise_i))
    for scale in [1]: # unbind:bind scale
        n_unbind = math.floor(n_bind * scale)
        x, y, class_name = read_processed_data(bind_count=n_bind, unbind_count=n_unbind, voxelise_i=voxelise_i)
        for repeat_i in range(n_repeat):
            logger.info("repeating {0}".format(repeat_i))
            x, y = shuffle(x, y)
            train_x, train_y, test_x, test_y = split_data(x, y)
            for model_i in [4]: # we select model 4
                model_name = model_names[model_i]
                model = get_model(model_i)
                file_name = 'box_size=%d,step=%d,epochs=%d,unbind=%d,model=%s,voxelise=%d,repeat=%d,train_ratio=0.8' % (
                     size, step, epochs, scale, model_name, voxelise_i, repeat_i)+',retrain=%d'
                checkpoint = ModelCheckpoint(model_dir+file_name%0+'_{epoch:02d}-loss={val_loss:.2f}-acc={val_acc:.2f}.h5', monitor='val_loss',
                                             verbose=2, save_best_only=True, save_weights_only=True, mode='auto',
                                             period=1)
                logger.info("*************** start training ****************")
                logger.info("model is {0}".format(model_name))
                logger.info("box size is {0}".format(size))
                logger.info("step is {0}".format(step))
                logger.info("epochs is {0}".format(epochs))
                logger.info("process from index {0} to {1}".format(1, processed_amount))
                logger.info("unbind:bind scale is {0}:1".format(scale))
                logger.info("training {0} bind data".format(n_bind))
                logger.info("training {0} unbind data".format(n_unbind))
                logger.info("voxelise is {0}".format(voxelise_i))

                print (model.summary())

                model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                h = model.fit(batch_size=32, x=x, y=y, epochs=epochs, verbose=2,
                              validation_split=0.2, callbacks=[earlystopper,checkpoint])
                save_model_info(file_name%0, model, h)
                for repeat_count in range(1, n_retrain+1):
                    train_x, train_y = shuffle(train_x, train_y)
                    logger.info('retrain {0}'.format(repeat_count))
                    # load model
                    loaded_model = load_model(file_name%(repeat_count-1))
                    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

                    logger.info('retraining...')
                    h = loaded_model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2,
                                         validation_data=(test_x, test_y), callbacks=[earlystopper, checkpoint])
                    save_model_info(file_name % repeat_count, loaded_model, h)

                logger.info("*************** end training ****************")


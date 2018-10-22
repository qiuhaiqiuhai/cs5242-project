from models import trial_3Dcnn as test0_network
from models import test1_3Dcnn as test1_network
from models import test2_3Dcnn as test2_network
from models import test3_3Dcnn as test3_network
from keras import optimizers, losses
from keras.models import load_model, model_from_json
from data_reader import read_processed_data
from sklearn.utils import shuffle
import logging, math
import sys, os, datetime
import numpy as np
import CONST

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
epochs = 15
input_shape = (size, size, size, 4)
processed_amount = CONST.DATA.processed_amount
n_bind = 10000
n_repeat = 0 # retrain how many times. If no need to retrain, put 0
selected_acc = 0.95

# define models
model_names = ['test0', 'test1', 'test2', 'test3']
models = []
models.append(test0_network(input_shape=input_shape))
models.append(test1_network(input_shape=input_shape))
models.append(test2_network(input_shape=input_shape))
models.append(test3_network(input_shape=input_shape))
optimizer = optimizers.adadelta()

def save_model_info(file_name, model, h):
    logger.info('save model weights in {0}...'.format(model_dir+file_name))
    model.save_weights(os.path.join(model_dir, file_name+'.h5'))
    model_json = model.to_json()
    with open(os.path.join(model_dir, file_name+'.json'), "w") as json_file:
        json_file.write(model_json)

    logger.info('save results in {0}...'.format(dir+file_name))
    np.savetxt(os.path.join(dir, file_name+ '.txt'), \
               np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))
    best_acc = max(h.history['val_acc'])
    if best_acc > selected_acc:
        logger.info('model acc > {1}, is saved as {0}...'.format(selected_dir+file_name, selected_acc))
        model.save_weights(os.path.join(selected_dir, file_name + '_%.2f.h5'%best_acc))
        with open(os.path.join(selected_dir, file_name + '_%.2f.json'%best_acc), "w") as json_file:
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


for scale in [1, 2, 3]:
    n_unbind = math.floor(n_bind * scale)

    x, y, class_name = read_processed_data(n_bind, n_unbind)
    x, y = shuffle(x, y)
    train_x, train_y, test_x, test_y = split_data(x, y)

    for i in [3]: # we select model 3
        model_name = model_names[i]
        model = models[i]
        file_name = 'box_size=%d,step=%d,epochs=%d,unbind=%d,model=%s' % (
		     size, step, epochs, scale, model_name)+',repeat=%d'
        logger.info("*************** start ****************")
        logger.info("model is {0}".format(model_name))
        logger.info("box size is {0}".format(size))
        logger.info("step is {0}".format(step))
        logger.info("epochs is {0}".format(epochs))
        logger.info("process from index {0} to {1}".format(1, processed_amount))
        logger.info("unbind:bind scale is {0}:1".format(scale))
        logger.info("training {0} bind data".format(n_bind))
        logger.info("training {0} unbind data".format(n_unbind))

        print (model.summary())

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        h = model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2, validation_data=(test_x, test_y))
        save_model_info(file_name%0, model, h)
        for repeat_count in range(1, n_repeat+1):
            train_x, train_y = shuffle(train_x, train_y)
            logger.info('repeat {0}'.format(repeat_count))
            # load model
            loaded_model = load_model(file_name%(repeat_count-1))
            loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            logger.info('retrain...')
            h = loaded_model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2, validation_data=(test_x, test_y))
            save_model_info(file_name % repeat_count, loaded_model, h)

        logger.info("*************** end ****************")


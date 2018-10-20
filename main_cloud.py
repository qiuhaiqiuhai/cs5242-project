from models import trial_3Dcnn as test0_network
from models import test1_3Dcnn as test1_network
from models import test2_3Dcnn as test2_network
from models import test3_3Dcnn as test3_network
from keras import optimizers, losses
from keras.models import load_model
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
if not os.path.exists(dir):
    os.makedirs(dir)
model_dir = 'models/%s/'%folder_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

size = CONST.VOXEL.size
step = CONST.VOXEL.step
epochs = 15
input_shape = (size, size, size, 4)
processed_amount = CONST.DATA.processed_amount
n_bind = 10000
n_repeat = 1 # retrain how many times. If no need to retrain, put 0

# define models
model_names = ['test0', 'test1', 'test2', 'test3']
models = []
models.append(test0_network(input_shape=input_shape))
models.append(test1_network(input_shape=input_shape))
models.append(test2_network(input_shape=input_shape))
models.append(test3_network(input_shape=input_shape))
optimizer = optimizers.adadelta()


for scale in [1, 2]:
	n_unbind = math.floor(n_bind * scale)

	train_x, train_y, class_name = read_processed_data(n_bind, n_unbind)

	for i in [3]: # we select model 3
		model_name = model_names[i]
		model = models[i]
		file_name = 'box_size=%d,step=%d,epochs=%d,unbind=%d,model=%s' % (
		     size, step, epochs, scale, model_name)+'_%d'

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
		h = model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2, validation_split=0.2)

		model.save(os.path.join(model_dir, file_name%0+'.h5'))

		np.savetxt(os.path.join(dir, file_name%0+'.txt'),\
		           np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))

		for repeat_count in range(1, n_repeat+1):
			train_x, train_y = shuffle(train_x, train_y)
			logger.info('repeat {0}'.format(i))

			loaded_model = load_model(os.path.join(model_dir, file_name%(repeat_count-1)+'.h5'))

			logger.info('retrain...')
			train_x, train_y = shuffle(train_x, train_y)
			h = loaded_model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2, validation_split=0.2)

			logger.info('save model in {0}...'.format(os.path.join(model_dir, file_name%repeat_count+'.h5')))
			loaded_model.save(
				os.path.join(model_dir, file_name%repeat_count+'.h5'))

			logger.info('save result in {0}...'.format(os.path.join(dir, file_name%repeat_count+'.txt')))
			np.savetxt(
				os.path.join(dir, file_name%repeat_count+'.txt'),
				np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))


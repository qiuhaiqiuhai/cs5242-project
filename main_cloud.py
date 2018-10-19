from models import trial_3Dcnn as test0_network
from models import test1_3Dcnn as test1_network
from models import test2_3Dcnn as test2_network
from models import test3_3Dcnn as test3_network
from keras import optimizers, losses
from data_reader import read_processed_data
from sklearn.utils import shuffle
import logging
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

size = CONST.VOXEL.size
step = CONST.VOXEL.step
epochs = 20
input_shape = (size, size, size, 4)
processed_amount = CONST.DATA.processed_amount
n_unbind = CONST.DATA.unbind_count
n_lig_atom = CONST.DATA.lig_data_max

# define models
model_names = ['test0', 'test1', 'test2', 'test3']
models = []
models.append(test0_network(input_shape=input_shape))
models.append(test1_network(input_shape=input_shape))
models.append(test2_network(input_shape=input_shape))
models.append(test3_network(input_shape=input_shape))
optimizer = optimizers.adadelta()

for n_unbind in [5, 10, 15]:
	for i in [0, 3]:
		model_name = model_names[i]
		model = models[i]

		logger.info("model is {0}".format(model_name))
		logger.info("box size is {0}".format(size))
		logger.info("step is {0}".format(step))
		logger.info("epochs is {0}".format(epochs))
		logger.info("process from index {0} to {1}".format(1, processed_amount))
		logger.info("max number of unbind pairs is {0}".format(n_unbind))
		logger.info("max number of atoms taken for each ligand is {0}".format(n_lig_atom))

		print (model.summary())

		train_x, train_y, class_name = read_processed_data(n_unbind)
		train_x, train_y = shuffle(train_x, train_y)

		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		h = model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2, validation_split=0.2)

		np.savetxt(os.path.join(dir,'box_size=%d,step=%d,epochs=%d,unbind=%d,model=%s.txt'%(size,step,epochs,n_unbind,model_name)),\
		           np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))
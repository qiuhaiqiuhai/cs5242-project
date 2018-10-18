from models import trial_3Dcnn as test_network
from models import test1_3Dcnn as test1_network
from keras import optimizers, losses
from data_reader import read_processed_data
from sklearn.utils import shuffle
import logging
import sys, os, datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

size = 18
step = 1
epochs = 20
input_shape = (size, size, size, 4)
min_index = 1
max_index = 3001
n_unbind = 1

logger = logging.getLogger('main')
logger.info("box size is {0}".format(size))
logger.info("step is {0}".format(step))
logger.info("epochs is {0}".format(epochs))
logger.info("process from index {0} to {1}".format(min_index, max_index))
logger.info("max number of unbind pairs is {0}".format(n_unbind))

train_x, train_y, class_name = read_processed_data(min_index, max_index, n_unbind)
train_x, train_y = shuffle(train_x, train_y)

model_name = 'test1'
model = test1_network(input_shape=input_shape)
optimizer = optimizers.adadelta()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
h = model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=1, validation_split=0.2)

date = datetime.datetime
folder_name = date.today().strftime('%Y-%m-%d_%H_%M_%S')
dir = 'results/%s/'%folder_name
if not os.path.exists(dir):
    os.makedirs(dir)
np.savetxt(os.path.join(dir,'results/box_size=%d,step=%d,epochs=%d,unbind=%d,model=%s.txt'%(size,step,epochs,n_unbind,model_name)),\
           np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))
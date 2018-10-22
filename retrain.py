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

logger = logging.getLogger('retrain')
date = datetime.datetime
folder_name = date.today().strftime('%Y-%m-%d')+'_retrain'
dir = 'results/%s/'%folder_name
if not os.path.exists(dir):
    os.makedirs(dir)
model_dir = 'models/%s/'%folder_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

epochs = 1
n_bind = 10000
n_unbind = 20000
n_repeat = 2
selected_model_dir = 'models/2018-10-20_22_29_00/box_size=19,step=1,epochs=1,unbind=1,model=test0_0.h5'

train_x, train_y, class_name = read_processed_data(n_bind, n_unbind)
train_x, train_y = shuffle(train_x, train_y)
loaded_model = load_model(selected_model_dir)
loss, acc = loaded_model.evaluate(train_x, train_y, verbose=2)

logger.info('original model on whole training set: loss={0}, acc={1}'.format(loss, acc))
logger.info('retrain {0} times'.format(n_repeat))
logger.info('epochs = {0}'.format(epochs))
logger.info('n_bind = {0}, n_unbind = {1}'.format(n_bind, n_unbind))
logger.info('retraining on model in '+selected_model_dir)
logger.info('original model structure')
print (loaded_model.summary())

for i in range(1, n_repeat+1):
	logger.info('repeat {0}'.format(i))
	if i > 1:
		loaded_model = load_model(os.path.join(model_dir,"model3_%d.h5" % (i-1)))

	logger.info('retrain...')
	train_x, train_y = shuffle(train_x, train_y)
	h = loaded_model.fit(batch_size=32, x=train_x, y=train_y, epochs=epochs, verbose=2, validation_split=0.2)

	logger.info('save model in {0}...'.format(os.path.join(model_dir, 'model3_%d' % (i))))
	loaded_model.save(
		os.path.join(model_dir, 'model3_%d.h5' % (i)))

	logger.info('save result in {0}...'.format(os.path.join(dir, 'model3_%d' % (i))))
	np.savetxt(
		os.path.join(dir,'model3_%d.txt'%(i)),
			   np.transpose([h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']]))



from models import trial_3Dcnn as test_network
from keras import optimizers, losses
from data_reader import read_processed_data
import CONST

size = CONST.VOXEL.size

input_shape = (size, size, size, 4)
train_x, train_y, test_x, test_y, class_name = read_processed_data()

model = test_network(input_shape=input_shape)
optimizer = optimizers.adadelta()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(batch_size=32, x=train_x, y=train_y, epochs=10, verbose=1)
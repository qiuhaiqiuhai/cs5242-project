import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv3D, Activation, MaxPool3D, Dropout, Flatten


def trial_3Dcnn(input_shape=(18, 18, 18, 4), class_num=2):
    """Example CNN

    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})

    Returns:
        model -- keras.models.Model() object
    """

    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3),
                     activation='relu', dilation_rate = 2,
                     input_shape=input_shape))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', dilation_rate = 2))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    return model


def test1_3Dcnn(input_shape=(18, 18, 18, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(64, input_shape=input_shape,kernel_size=(3, 3, 3), activation='relu', dilation_rate = 2))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, kernel_size=(2, 2, 2), activation='relu',dilation_rate = 2))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(256,kernel_size=(1, 1, 1), activation='relu',dilation_rate = 2))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    return model

def test2_3Dcnn(input_shape=(18, 18, 18, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(32, input_shape=input_shape,kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, kernel_size=(2, 2, 2), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(256,kernel_size=(1, 1, 1), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    return model

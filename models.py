import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv3D, Activation, MaxPool3D, Dropout, Flatten, BatchNormalization


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

# Development and evaluation of a deep learning model for protein–ligand binding affinity prediction
# this network use 21 as cubic size, min-batch = 5, input-features = 19
def test1_3Dcnn(input_shape=(21, 21, 21, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(64, input_shape=input_shape,kernel_size=(5, 5, 5), activation='relu'))
    model.add(Conv3D(128, kernel_size=(5, 5, 5), activation='relu'))
    model.add(Conv3D(256,kernel_size=(5, 5, 5), activation='relu'))
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

# DeepSite: protein-binding site predictor using 3D-convolutional neural networks
# this network use 16 as cubic size
def test2_3Dcnn(input_shape=(16, 16, 16, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(32, input_shape=input_shape,kernel_size=(3, 3, 3), activation='elu'))
    model.add(Conv3D(48, kernel_size=(3, 3, 3), activation='elu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='elu'))
    model.add(Conv3D(96,kernel_size=(3, 3, 3), activation='elu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='sigmoid'))

    return model

# AtomNet: A Deep Convolutional Neural Network for Bioactivity Prediction in Structure-based Drug Discovery
# this network use 20 as cubic size, min-batch = 768 samples
# can reach to 0.95 after 5-8 epochs
def test3_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(128, input_shape=input_shape,kernel_size=(5, 5, 5), activation='relu'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(256, kernel_size=(1, 1, 1), activation='relu'))
    model.add(Conv3D(256, kernel_size=(1, 1, 1), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))

    return model

# change padding to "same" and all kernal size as 3, remove one layer because with 4 layer, one epoch needs 555s
# time taken for one epoch is 463s
# best acc is around 0.955
# def test4_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
#     model = Sequential()
#     model.add(Conv3D(128, input_shape=input_shape,kernel_size=(3, 3, 3), activation='relu', padding="same"))
#     model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding="same"))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding="same"))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(class_num, activation='softmax'))

#     return model

# reduce filter numbers
# time taken for one epoch is 185s
# best acc is 0.949
# def test4_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
#     model = Sequential()
#     model.add(Conv3D(64, input_shape=input_shape,kernel_size=(3, 3, 3), activation='relu', padding="same"))
#     model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding="same"))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding="same"))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(class_num, activation='softmax'))

#     return model

# change kernel size to 5
# time taken for one epoch is 342s
# best acc is around 0.965
# def test4_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
#     model = Sequential()
#     model.add(Conv3D(64, input_shape=input_shape,kernel_size=(5, 5, 5), activation='relu', padding="same"))
#     model.add(Conv3D(128, kernel_size=(5, 5, 5), activation='relu', padding="same"))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv3D(256, kernel_size=(5, 5, 5), activation='relu', padding="same"))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dense(class_num, activation='softmax'))

#     return model

# remove padding for 2,3 3D conv layers reduce training params and add batch normalization before maxpool
# doesn't work
# def test4_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
#     model = Sequential()
#     model.add(Conv3D(64, input_shape=input_shape,kernel_size=(5, 5, 5), activation='relu', padding="same"))
#     model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.25))
#     model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
#     model.add(MaxPool3D(pool_size=(2, 2, 2)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1024, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(class_num, activation='softmax'))

#     return model

# remove padding for 2,3 3D conv layers reduce training params and remove one dense layer
# time taken for one epoch is 129s
# best acc is around 0.963 at epoch 6
def test4_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(64, input_shape=input_shape,kernel_size=(5, 5, 5), activation='relu', padding="same"))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    return model


def test5_3Dcnn(input_shape=(20, 20, 20, 4), class_num=2):
    model = Sequential()
    model.add(Conv3D(64, input_shape=input_shape,kernel_size=(5, 5, 5), activation='relu', padding="same", dilation_rate=2))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', dilation_rate=2))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', dilation_rate=2))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))

    return model


if __name__ == '__main__':
    # 8,219,330 params
    model = test4_3Dcnn(input_shape=(19, 19, 19, 4))
    print(model.summary())

    # 17,918,658 params
    model = test4_3Dcnn(input_shape=(25, 25, 25, 4))
    print (model.summary())

    # 8,219,330 params
    model = test5_3Dcnn(input_shape=(25, 25, 25, 4))
    print(model.summary())


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, ZeroPadding2D, 
                                     Dropout, Flatten, Dense, BatchNormalization)
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

def create_base_network(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(ZeroPadding2D(padding=2))

    model.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer=GlorotUniform()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.3))
    model.add(ZeroPadding2D(padding=1))

    model.add(Conv2D(384, (3, 3), activation='relu', padding='same', kernel_initializer=GlorotUniform()))
    model.add(ZeroPadding2D(padding=1))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=GlorotUniform()))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.0005), kernel_initializer=GlorotUniform()))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0005), kernel_initializer=GlorotUniform()))
    return model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPool2D
from keras import regularizers

# Constants
GridW = GridH = 7
NumBoundingBoxes = 2
Classes = ["Sphere", "Can", "Bottle"]
BoundingBoxOverhead = 5 # 4 for x,y,w,h and 1 for confidence


def getNetwork():
    reg = regularizers.l2()

    model = Sequential()
    model.add(Convolution2D(64, (7,7), 2, padding="same", input_shape=(448,448,3), kernel_regularizer=reg))
    model.add(MaxPool2D((2,2), 2, padding="same"))

    model.add(Convolution2D(192, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(MaxPool2D((2,2), 2, padding="same"))

    model.add(Convolution2D(128, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(256, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(256, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(512, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(MaxPool2D((2,2), 2, padding="same"))

    model.add(Convolution2D(256, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(512, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(256, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(512, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(256, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(512, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(256, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(512, (3,3), 1, padding="same", kernel_regularizer=reg))

    model.add(Convolution2D(512, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(1024, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(MaxPool2D((2,2), 2, padding="same"))

    model.add(Convolution2D(512, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(1024, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(512, (1,1), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(1024, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(1024, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(1024, (3,3), 2, padding="same", kernel_regularizer=reg))
    
    model.add(Convolution2D(1024, (3,3), 1, padding="same", kernel_regularizer=reg))
    model.add(Convolution2D(1024, (3,3), 1, padding="same", kernel_regularizer=reg))

    model.add(Flatten())
    model.add(Dense(4096, activation="softmax"))
    #TODO: Add dropout
    model.add(Dense(GridW*GridH+(NumBoundingBoxes*BoundingBoxOverhead+ len(Classes)), activation="softmax"))
    model.add(Reshape([GridW,GridH, (NumBoundingBoxes*BoundingBoxOverhead) + len(Classes)]))

    return model
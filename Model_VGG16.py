from keras.models import Sequential
from keras.layers import *
from keras import backend as K
import cv2 as cv, numpy as np
from keras.src.layers import Conv2D


def VGG_16(weights_path=None, num_of_class=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=num_of_class, activation="softmax"))
    return model


def Model_VGG16(train_data, train_target):
    print('VGG16')
    ## VGG16
    IMG_SIZE = [32, 32, 3]

    # Resize train_data to (32, 32, 3)
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i] = cv.resize(train_data[i], (IMG_SIZE[1], IMG_SIZE[0]))

    model = VGG_16(num_of_class=train_target.shape[1])
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(x=Feat1, y=train_target, epochs=5, steps_per_epoch=5, batch_size=4)

    pred = model.predict(Feat1)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # using Keras backend function

    layerNo = 1
    Feats = []
    for i in range(train_data.shape[0]):
        test = Feat1[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)

    return Feats

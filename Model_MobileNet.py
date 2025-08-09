import cv2 as cv
import numpy as np
from PIL import Image
from keras import backend as K
from keras.applications import MobileNet
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image


def Model_MobileNet(train_Data, Train_Target):
    model = MobileNet(weights='imagenet')
    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for s in range(train_Data.shape[0]):
        Feat[s, :] = cv.resize(train_Data[s], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))[:, :, 0]
    Train_Data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    for k in range(Train_Data.shape[0]):
        data = Image.fromarray(np.uint8(Train_Data[k])).convert('RGB')
        data = image.img_to_array(data)
        data = np.expand_dims(data, axis=0)
        data = np.squeeze(data)
        Train_Data[k] = cv.resize(data, (224, 224))
        Train_Data[k] = preprocess_input(Train_Data[k])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])  # optimizer='Adam'
    model.fit_generator(Train_Data, Train_Target, epochs=10, steps_per_epoch=5, batch_size=4)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # using Keras backend function
    layerNo = 1
    Feats = []
    for i in range(train_Data.shape[0]):
        print(i)
        test = Train_Data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    return Feats

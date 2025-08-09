import cv2 as cv
from keras import backend as K
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras_applications.resnet_common import ResNet50
from keras import Sequential
import numpy as np
import tensorflow as tf
from keras.src.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from Classificaltion_Evaluation import ClassificationEvaluation


def Inception(Data, Target):
    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']
    # Define the input shape
    input_shape = (124, 124, 3)  # Adjust the shape based on your data
    num_classes = Target.shape[-1]

    # Adjust Feat1 to have an extra dimension for color channels
    Feat1 = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Data.shape[0]):
        Feat1[i, :] = cv.resize(Data[i], (input_shape[1], input_shape[0]))
    train_data = Feat1

    # Load the InceptionV3 model (pre-trained on ImageNet)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='linear')(x)  # 'relu'
    predictions = Dense(num_classes, activation='softmax')(x)  # 'softmax'
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Fit the model
    model.fit(train_data, Target, epochs=5, batch_size=5, steps_per_epoch=5)
    pred = model.predict(train_data)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # using Keras backend function
    layerNo = 1
    Feature = []
    for i in range(Data.shape[0]):
        print(i)
        test = train_data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feature.append(layer_out)
    return Feature


# Ensure image data format is set explicitly
tf.keras.backend.set_image_data_format('channels_last')


def Resnet50(Data, Target):
    input_shape = (224, 224, 3)
    Feat1 = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))

    # Resize images
    for i in range(Data.shape[0]):
        Feat1[i, :] = cv.resize(Data[i], (input_shape[1], input_shape[0]))

    train_data = Feat1

    # Build the ResNet50-based model
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')
    x = base_model.output
    x = Dense(units=5, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    model.fit(train_data, Target, epochs=5, batch_size=4)

    # Extract features from a specific layer
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [tf.keras.backend.function([inp], [out]) for out in outputs]
    layerNo = 1
    Feature = []
    for i in range(Data.shape[0]):
        test = train_data[i, :, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feature.append(layer_out)

    return Feature


def Model_IR_CNN(data, Target):
    Feat_1 = Inception(data, Target)
    Feat_2 = Resnet50(data, Target)
    Feat = np.concatenate((Feat_1, Feat_2), axis=1)

    per = round(Feat.shape[0] * 0.75)
    train_data = Feat[:per, :]
    train_target = Target[:per, :]
    test_data = Feat[per:, :]
    test_target = Target[per:, :]
    print('CNN')
    IMG_SIZE = 28
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    X_train = Train_X.astype('float32') / 255
    X_test = Test_X.astype('float32') / 255
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_target.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, train_target, epochs=3, batch_size=64, validation_split=0.1)
    pred = model.predict(X_test)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = ClassificationEvaluation(test_target.astype('int'), test_target)
    return Eval

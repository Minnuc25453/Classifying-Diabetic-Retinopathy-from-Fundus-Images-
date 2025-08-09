import numpy as np
from keras import Sequential
from keras.src.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_CNN(train_data, train_target, test_data, test_target):
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


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


def Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target, soln=None):
    sol = [5, 5]

    # Ensure the model output matches the target shape
    num_classes = Train_Target.shape[1]

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=sol[0], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=sol[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

    # Initialize prediction array
    pred = np.zeros((Test_Target.shape[0], num_classes))

    for i in range(num_classes):
        print(f"Training model for class {i + 1}/{num_classes}")
        model.fit(Train_Data, Train_Target, epochs=2, batch_size=16, validation_split=0.1)
        pred = model.predict(Test_Data).flatten()

    # Threshold predictions for binary classification
    pred[pred >= 0.8] = 1
    pred[pred < 0.8] = 0

    # Evaluate accuracy
    Eval = accuracy_score(Test_Target, pred)
    return Eval



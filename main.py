import glob

import numpy as np
import pydicom as dicom
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

from hog import extract_carac

path_bmt = 'CINTILOGRAFIAS/BMT'
path_graves = 'CINTILOGRAFIAS/GRAVES'

if __name__ == '__main__':

    bmp_images = []
    graves_images = []
    labels = []
    all = []

    for file in glob.glob(path_bmt + '/**/*.dcm'):
        ds = dicom.dcmread(file)
        pixel_array_numpy = ds.pixel_array
        bmp_images.append(pixel_array_numpy)

    for file in glob.glob(path_graves + '/**/*.dcm'):
        ds = dicom.dcmread(file)
        pixel_array_numpy = ds.pixel_array
        graves_images.append(pixel_array_numpy)

    bmp_images = extract_carac(bmp_images)
    graves_images = extract_carac(graves_images)

    labels = []
    for file in bmp_images:
        file = file.reshape(128 * 128)
        all.append(file)
        labels.append(0)

    for file in graves_images:
        file = file.reshape(128 * 128)
        all.append(file)
        labels.append(1)

    X = all
    X = np.asarray(X)
    Y = labels
    Y = np.asarray(Y)
    loo = LeaveOneOut()

    i = 0
    for train_index, test_index in loo.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        model = Sequential()
        model.add(Dense(20, input_shape=(16384,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  epochs=150, batch_size=10,
                  callbacks=early_stop,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save('modelos/modelo_' + str(i))
        i = i + 1

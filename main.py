import glob

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from sklearn.model_selection import train_test_split
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

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.08, random_state=9)

    print('Qtde de treino: {}'.format(len(x_train)))
    print('Qtde de validação: {}'.format(len(x_val)))

    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    model = Sequential()
    model.add(Dense(10, input_shape=(16384,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=150, batch_size=10,
                        callbacks=early_stop,
                        validation_data=(x_val, y_val))

    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    plt.show()

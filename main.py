import glob
import pydicom as dicom
import numpy as np
import pandas as pd

from hog import extract_carac
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

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


    for file in bmp_images:
        file = file.reshape(128 * 128)
        all.append([file, 0])
    
    for file in graves_images:
        file = file.reshape(128 * 128)
        all.append([file, 1])

    df = pd.DataFrame(all)

    X = df[0]
    Y = df[1]

    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=9)
    print('Qtde de treino: {}'.format(len(x_train)))
    print('Qtde de validação: {}'.format(len(x_val)))

    print('-----------------------------------------------------------------')
    print('X = ')
    print(X)
    print('-----------------------------------------------------------------')
    print('y = ')
    print(Y)
    print('-----------------------------------------------------------------')

    model = Sequential()
    model.add(Dense(30, input_shape=(128,128), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=150, batch_size=10)

    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))
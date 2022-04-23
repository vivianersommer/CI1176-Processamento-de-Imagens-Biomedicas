import glob
import pydicom as dicom
import numpy as np
import pandas as pd

from hog import extract_carac
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
        all.append([file, 0])
    
    for file in graves_images:
        all.append([file, 1])

    df = pd.DataFrame(all)

    X = df[:-1]
    y = df[-1:]

    x_images = X[0]
    x_labels = X[1]

    y_images = y[0]
    y_labels = y[1]

    x_all = x_images + x_labels
    y_all = y_images + y_labels

    print('-----------------------------------------------------------------')
    print('X = ')
    print(X)
    print('-----------------------------------------------------------------')
    print('y = ')
    print(y)
    print('-----------------------------------------------------------------')

    model = Sequential()
    model.add(Dense(x_images[0].shape[1], input_dim=x_images[0].shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_images, y_images, epochs=150, batch_size=10)

    # _, accuracy = model.evaluate(X, y)
    # print('Accuracy: %.2f' % (accuracy*100))
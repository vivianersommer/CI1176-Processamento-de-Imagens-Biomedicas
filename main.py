import glob

import numpy as np
import pydicom as dicom
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix


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
    resultado_original = []
    resultado_obtido = []
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
                  validation_data=(x_test, y_test),
                  verbose = False)

        predictions = (model.predict(x_test) > 0.5).astype("int32")
        resultado_original.append(float(y_test[0]))
        resultado_obtido.append(predictions[0][0])
        cf_matrix = confusion_matrix([float(y_test[0])], [predictions[0][0]])
        print("Modelo " + str(i))
        print(cf_matrix)
        i = i + 1

    print(resultado_original)
    print(resultado_obtido)

    cf_matrix = confusion_matrix(resultado_original, resultado_obtido)
    print(cf_matrix)
    #
    # tn, fp, fn, tp = cf_matrix
    #
    # # Sensibilidade
    # # VP / (VP+FN)
    # sensibilidade = tp / (tp + fn)
    #
    # # Especificidade
    # # VN / (FP+VN)
    # especifidade = tp / (fp + tn)
    #
    # # Acurácia
    # # (VP+VN) / N
    # acuracia = (tp + tn) / (fp + fn)
    #
    # # Precisão
    # # VP / (VP+FP)
    # precisao = tp / (tp + fp)

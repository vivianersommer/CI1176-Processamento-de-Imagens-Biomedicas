import glob
import logging

import numpy as np
import pydicom as dicom
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

path_bmt = 'CINTILOGRAFIAS/BMT'
path_graves = 'CINTILOGRAFIAS/GRAVES'


def ler_imagens():
    bmp_images = []
    graves_images = []

    for file in glob.glob(path_bmt + '/**/*.dcm'):
        ds = dicom.dcmread(file)
        pixel_array_numpy = ds.pixel_array
        bmp_images.append(pixel_array_numpy)

    for file in glob.glob(path_graves + '/**/*.dcm'):
        ds = dicom.dcmread(file)
        pixel_array_numpy = ds.pixel_array
        graves_images.append(pixel_array_numpy)

    return bmp_images, graves_images


def extract_carac(images):
    images_hog = []
    for image in images:
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)

        images_hog.append(hog_image)
    return images_hog


def extracao_caracteristicas(bmp_images, graves_images):
    bmp_images = extract_carac(bmp_images)
    graves_images = extract_carac(graves_images)

    return bmp_images, graves_images


def criar_labels(bmp_images, graves_images):
    labels = []
    all = []

    for file in bmp_images:
        # alinhamento da imagem em uma linha
        file = file.reshape(128 * 128)

        all.append(file)
        labels.append(0)

    for file in graves_images:
        # alinhamento da imagem em uma linha
        file = file.reshape(128 * 128)

        all.append(file)
        labels.append(1)

    return labels, all


def criar_modelos(labels, all_images):
    X = all_images
    X = np.asarray(X)
    Y = labels
    Y = np.asarray(Y)
    loo = LeaveOneOut()

    i = 1
    resultado_original = []
    resultado_obtido = []
    tn = []
    fp = []
    fn = []
    tp = []

    print("Gerando os modelos ----------------------------------------------------------------------------------------")
    for train_index, test_index in loo.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        model = Sequential()
        model.add(Dense(30, input_shape=(16384,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  epochs=50, batch_size=10,
                  callbacks=early_stop,
                  validation_data=(x_test, y_test),
                  verbose=False)

        predictions = (model.predict(x_test) > 0.5).astype("int32")
        resultado_original.append((y_test[0]).astype("int32"))
        resultado_obtido.append(predictions[0][0])

        cf_matrix = confusion_matrix([y_test[0]], [predictions[0][0]], labels=[True, False])
        tn_y, fp_y, fn_y, tp_y = cf_matrix.ravel()
        tn.append(tn_y)
        fp.append(fp_y)
        fn.append(fn_y)
        tp.append(tp_y)
        i = i + 1

    return resultado_original, resultado_obtido, tn, fp, fn, tp


def analisar_matriz_confusao(resultado_original, resultado_obtido, tn, fp, fn, tp):
    i = 0
    for val in tn:
        if resultado_original[0] == 0:
            label_result = "BMT"
        else:
            label_result = "GRAVES"

        print("Modelo " + str(i + 1) + " - Confusion Matrix ---------------------------------")
        print("Imagem de avaliação = " + label_result + " - " + str(resultado_original[0]))
        print(" [ " + str(tp[i]) + "   " + str(fp[i]) + " ] ")
        print(" [ " + str(fn[i]) + "   " + str(tn[i]) + " ] ")
        print("-------------------------------------------------------------")
        i = i + 1

    tn_y, fp_y, fn_y, tp_y = confusion_matrix(resultado_original, resultado_obtido).ravel()
    print("Todos os modelos - Confusion Matrix --------------------------")
    print(" [ " + str(tp_y) + "   " + str(fp_y) + " ] ")
    print(" [ " + str(fn_y) + "   " + str(tn_y) + " ] ")
    print("-------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------------------------------")


def analisar_metricas(tn, fp, fn, tp):
    print("Avaliando os modelos --------------------------------------------------------------------------------------")
    sensibilidade = []
    especifidade = []
    acuracia = []

    # Sensibilidade
    for i, val in enumerate(tp):
        if (tp[i] + fn[i]) == 0:
            continue
        else:
            sens = tp[i] / (tp[i] + fn[i])
        sensibilidade.append(sens)

    # Especificidade
    for i, val in enumerate(tp):
        if (tn[i] + fp[i]) == 0:
            continue
        else:
            espec = tn[i] / (tn[i] + fp[i])
        especifidade.append(espec)

    # Acurácia
    for i, val in enumerate(tp):
        if (fp[i] + fn[i] + tp[i] + tn[i]) == 0:
            continue
        else:
            acur = (tp + tn[i]) / (fp[i] + fn[i] + tp[i] + tn[i])
        acuracia.append(acur)

    print("Sensibilidade - Geral - Média -------------------------------")
    print(np.mean(sensibilidade))
    print("-------------------------------------------------------------")

    print("Sensibilidade - Geral - Desvio Padrão -----------------------")
    print(np.std(sensibilidade))
    print("-------------------------------------------------------------")

    print("Especificidade - Geral - Média ------------------------------")
    print(np.mean(especifidade))
    print("-------------------------------------------------------------")

    print("Especificidade - Geral - Desvio Padrão ----------------------")
    print(np.std(especifidade))
    print("-------------------------------------------------------------")

    print("Acurácia - Geral - Média ------------------------------------")
    print(np.mean(acuracia))
    print("-------------------------------------------------------------")

    print("Acurácia - Geral - Desvio Padrão ----------------------------")
    print(np.std(acuracia))
    print("-------------------------------------------------------------")


if __name__ == '__main__':
    logging.getLogger('tensorflow').disabled = True  # evita alguns warnings do tensorflow, por nao ter GPU

    bmp_images, graves_images = ler_imagens()

    bmp_images, graves_images = extracao_caracteristicas(bmp_images, graves_images)

    labels, all_images = criar_labels(bmp_images, graves_images)

    resultado_original, resultado_obtido, tn, fp, fn, tp = criar_modelos(labels, all_images)

    analisar_matriz_confusao(resultado_original, resultado_obtido, tn, fp, fn, tp)

    analisar_metricas(tn, fp, fn, tp)

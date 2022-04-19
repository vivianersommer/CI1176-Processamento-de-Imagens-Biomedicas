# Imports

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, \
    ModelCheckpoint

from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



# Functions

def train_model(X_train, y_train, ratio=0.0, weight_ratio=1.0, pca_ratio=0.0):

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    # Undersample
    X_train, X_val, y_train, y_val = undersample(X_train, X_val, y_train, y_val, ratio)

    # Filter outliers
    X_train, y_train = remove_outliers(X_train, y_train, pca_ratio)

    # Training parameters
    class_weight = {
        False: 1.0, 
        True: weight_ratio*(np.sum(~y_train)/np.sum(y_train))
    }
    early_stop = EarlyStopping(monitor='val_loss', patience=30)
    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)

    # Create model
    model = create_model(X_train)

    # Fit model
    model.fit(X_train, y_train, \
        validation_data=(X_val, y_val), batch_size=32, \
        epochs=200, class_weight=class_weight, \
        callbacks=[early_stop, checkpointer])
    model.load_weights(model_name)

    return model



def create_model(X_train):

    model = Sequential()
    model.add(Dense(X_train.shape[1], \
        input_dim=X_train.shape[1], \
        activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', \
        optimizer='adam', \
        metrics='accuracy')

    return model



def remove_outliers(X_train, y_train, pca_ratio=0.0):

    mask = np.ones(y_train.shape) == 1

    if pca_ratio > 0.0:

        pca = PCA(n_components=1).fit(X_train[y_train])
        X_pca = pca.inverse_transform(\
            pca.transform(X_train[y_train]))
        mse = ((X_train[y_train] - X_pca)**2).sum(axis=1)
        mask[y_train] = (mse <= np.quantile(mse, pca_ratio))

        pca = PCA(n_components=1).fit(X_train[~y_train])
        X_pca = pca.inverse_transform(\
            pca.transform(X_train[~y_train]))
        mse = ((X_train[~y_train] - X_pca)**2).sum(axis=1)
        mask[~y_train] = (mse <= np.quantile(mse, 1.0))

    return X_train[mask], y_train[mask]



def undersample(X_train, X_val, y_train, y_val, ratio=0.0):

    if ratio > 0.0:
        under_sampler = RandomUnderSampler(ratio, random_state=42)
        X_train, y_train = \
            under_sampler.fit_resample(X_train, y_train)
        # X_val, y_val = under_sampler.fit_resample(X_val, y_val)

    return X_train, X_val, y_train, y_val



def convert_mlp_tf2np(path):

    # Load tf model
    model = tf.keras.models.load_model(path)

    # Get model arrays
    hidden_weights = model.layers[0].weights[0].numpy()
    hidden_bias = model.layers[0].weights[1].numpy()
    output_weights = model.layers[2].weights[0].numpy()
    output_bias = model.layers[2].weights[1].numpy()

    # Save model
    new_path = '.'.join(path.split('.')[:-1]) + '.npz'
    np.savez(new_path, \
        hidden_weights=hidden_weights, \
        hidden_bias=hidden_bias, \
        output_weights=output_weights, \
        output_bias=output_bias)

    return new_path
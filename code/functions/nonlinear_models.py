import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def NN_forecast(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    # model with 3 hidden layers and 100 neurons each with dropout

    # determine the number of features (columns) in X_train
    num_features = X_train.shape[1]

    # create a layer after the other in a sequential way
    NN_model = Sequential()
    # adding first layer: units is the number of neurons, activation is the function, input_dim: number of features
    NN_model.add(Dense(units=100, activation='relu', input_dim=num_features))
    # adding a dropout in the first layer to the second layer
    NN_model.add(Dropout(0.2))
    # adding second layer: units is the number of neurons, activation is the function
    NN_model.add(Dense(units=60, activation='relu'))
    # adding a dropout in the second layer to the third layer
    NN_model.add(Dropout(0.2))
    # adding third layer: units is the number of neurons, activation is the function
    NN_model.add(Dense(units=30, activation='relu'))
    # adding last layer: output layer
    NN_model.add(Dense(units=1, activation='linear'))

    # compiling the model
    NN_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # standarize the data
    Xtrain = (X_train - X_train.mean(axis=0))/X_train.std(axis=0)
    Xtest = (X_test - X_train.mean(axis=0))/X_train.std(axis=0)

    # epochs: the number of times the model will pass through the whole train data, batch_size: the number of rows the model will be trained
    result = NN_model.fit(Xtrain, y_train, epochs=1000, batch_size=32, verbose=0)

    y_pred = NN_model.predict(Xtest, verbose = 0).ravel()
    error_pred = (y_test - y_pred)**2

    return(y_pred, error_pred)



def LSTM_forecast(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    # model with 3 hidden layer and 50 neurons and dropout

    # determine the number of features (columns) in X_train
    num_features = X_train.shape[1]

    # reshape X_train and X_test to have the shape (num_samples, timesteps, num_features)
    X_train = np.array(X_train).reshape(-1, 1, num_features)
    X_test = np.array(X_test).reshape(-1, 1, num_features)

    # create a layer after other in a sequential way
    LSTM_model = Sequential()
    # adding first layer: unit is the number of neurons, activation is the function, input_dim: number of predictors
    LSTM_model.add(LSTM(units=100, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape=(1, num_features)))
    # adding a dropout in the first layer to second layer
    LSTM_model.add(Dropout(0.2))
    # adding second layer: unit is the number of neurons, activation is the function, input_dim: number of predictors
    LSTM_model.add(LSTM(units=60, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    # adding a dropout in the first layer to second layer
    LSTM_model.add(Dropout(0.2))
    # adding fourth layer: unit is the number of neurons, activation is the function, input_dim: number of predictors
    LSTM_model.add(LSTM(units=30, activation='tanh', recurrent_activation='sigmoid'))
    # adding last layer: output layer
    LSTM_model.add(Dense(units=1, activation='linear'))

    # compiling the model
    LSTM_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    # standarize the data
    Xtrain = (X_train - X_train.mean(axis=0))/X_train.std(axis=0)
    Xtest = (X_test - X_train.mean(axis=0))/X_train.std(axis=0)

    result = LSTM_model.fit(Xtrain, y_train, epochs=1000, batch_size=32, verbose=0)

    y_pred = LSTM_model.predict(Xtest, verbose = 0).ravel()
    error_pred = (y_test - y_pred)**2

    return(y_pred, error_pred)
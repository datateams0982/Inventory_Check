import numpy as np # linear algebra
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import pickle
import math

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, TimeDistributed, LeakyReLU, Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

def validation_split(X_train, Y_train, step=20):
    length = len(X_train)//step
    val_index = [(step * i) - 1 for i in range(1, length + 1, 1)]
    train_index = [i for i in range(len(X_train)) if i%step != step-1]
    valX = [X_train[i] for i in val_index]
    trainX = [X_train[i] for i in train_index]
    valY = [Y_train[i] for i in val_index]
    trainY = [Y_train[i] for i in train_index]
    
    return trainX, trainY, valX, valY   


def get_dependent_variable(Y_train, Y_val, Y_test):
        training = [item[2] for item in Y_train]
        val = [item[2] for item in Y_val]
        testing = [item[2] for item in Y_test]

        return training, val, testing


def categorical_transform(data):

    onehot_encoder = OneHotEncoder(sparse=False)

    d = np.array(data).reshape(len(data), 1)
    Y = onehot_encoder.fit_transform(d)

    return Y



def Separate_Evaluation(Feature, Real, prediction, model, **kwargs):
        
        r = prediction.tolist()
        y_pred = []
        for i, item in enumerate(r):
                y_pred.append(item.index(max(item))) 
                
        
        if len(kwargs.keys()) == 1:
                if 'year' in kwargs:
                        value = kwargs['year']
                        y_true = [sublist[2] for sublist in Real if sublist[0].year == value]  
                        y_index = [i for i, sublist in enumerate(Real) if sublist[0].year == value]
                else:
                        value = kwargs['stock']
                        y_true = [sublist[2] for sublist in Real if sublist[1] == value]  
                        y_index = [i for i, sublist in enumerate(Real) if sublist[1] == value]

        else:
                year = kwargs['year']
                stock = kwargs['stock']
                y_true = [sublist[2] for sublist in Real if (sublist[0].year == year) and (sublist[1] == stock)]  
                y_index = [i for i, sublist in enumerate(Real) if (sublist[0].year == year) and (sublist[1] == stock)]

        
        y_true_encode = categorical_transform(y_true)
        y = [y_pred[i] for i in y_index]
        X = [Feature[i] for i in y_index]
        
        correct = 0
        for i, item in enumerate(y):
            if item == y_true[i]:
                correct += 1
            else:
                continue
                
        target_names = ['down', 'up']
        auc = model.evaluate(np.array(X), y_true_encode)[1]
        
        report = classification_report(y_true, y, target_names=target_names)
        accuracy = correct/len(y)
        

        return report, accuracy, auc



def LSTM(X_train, Y_train, hyperparameters_LSTM, problem='regression'):
        model = Sequential()
        model.add(LSTM(
                name='lstm_0',
                units=hyperparameters_LSTM["l1_out"],
                return_sequences=True,
                stateful=False,
                dropout=hyperparameters_LSTM["l1_drop"],
                recurrent_dropout=hyperparameters_LSTM["l1_drop"],
                activation='relu'))

        if hyperparameters_LSTM['layers'] == 2:
                model.add(LSTM(
                        name='lstm_1',
                        units=hyperparameters_LSTM["l2_out"],
                        return_sequences=False,
                        stateful=False,
                        dropout=hyperparameters_LSTM["l2_drop"],
                        recurrent_dropout=hyperparameters_LSTM["l2_drop"],
                        activation='relu'))

        elif hyperparameters_LSTM['layers'] == 3:
                model.add(LSTM(
                        name='lstm_1',
                        units=hyperparameters_LSTM["l2_out"],
                        return_sequences=True,
                        stateful=False,
                        dropout=hyperparameters_LSTM["l2_drop"],
                        recurrent_dropout=hyperparameters_LSTM["l2_drop"],
                        activation='relu'))

                model.add(LSTM(
                        name='lstm_2',
                        units=hyperparameters_LSTM["l3_out"],
                        return_sequences=False,
                        stateful=False,
                        dropout=hyperparameters_LSTM["l3_drop"],
                        recurrent_dropout=hyperparameters_LSTM["l3_drop"],
                        activation='relu'))

        else:
                model.add(LSTM(
                        name='lstm_1',
                        units=hyperparameters_LSTM["l2_out"],
                        return_sequences=True,
                        stateful=False,
                        dropout=hyperparameters_LSTM["l2_drop"],
                        recurrent_dropout=hyperparameters_LSTM["l2_drop"],
                        activation='relu'))

                model.add(LSTM(
                        name='lstm_2',
                        units=hyperparameters_LSTM["l3_out"],
                        return_sequences=True,
                        stateful=False,
                        dropout=hyperparameters_LSTM["l3_drop"],
                        recurrent_dropout=hyperparameters_LSTM["l3_drop"],
                        activation='relu'))
        
                model.add(LSTM(
                        name='lstm_3',
                        units=hyperparameters_LSTM["l4_out"],
                        return_sequences=True,
                        stateful=False,
                        dropout=hyperparameters_LSTM["l4_drop"],
                        recurrent_dropout=hyperparameters_LSTM["l4_drop"],
                        activation='relu'))

        if problem == 'regression':
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_squared_error',
                        optimizer=optimizers.Nadam(lr=0.001),
                        metrics=['mape'])

        else:
                model.add(Dense(3, activation='softmax'))
                model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Nadam(lr=0.001),
                        metrics=['accuracy'])
    
        early_stopping = EarlyStopping(monitor='val_loss', patience=hyperparameters_LSTM["patience"], verbose=1)
        model.fit(np.array(X_train), np.array(Y_train), validation_split=0.05, batch_size = hyperparameters_LSTM["batch_size"], epochs=hyperparameters_LSTM["epochs"], verbose=2, callbacks=[early_stopping])

        return model


def CNN(X_train, Y_train, hyperparameters_CNN, problem='regression'):
        model = Sequential()
        model.add(Conv1D(
                filters=hyperparameters_CNN['l1_filter'], 
                kernel_size=hyperparameters_CNN['l1_kernel'], 
                activation='relu', 
                input_shape=(np.array(X_train).shape[1], np.array(X_train).shape[2]), 
                padding=hyperparameters_CNN['padding']))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(
                filters=hyperparameters_CNN['l2_filter'], 
                kernel_size=hyperparameters_CNN['l2_kernel'], 
                activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(
                filters=hyperparameters_CNN['l3_filter'], 
                kernel_size=hyperparameters_CNN['l3_kernel'], 
                activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dropout(0.2)) 
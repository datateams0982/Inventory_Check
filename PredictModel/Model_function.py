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


def transform_back(array, cluster, scaler='standardize'):
        filename = 'D:\\庫存健診開發\\data\\Training\\Raw\\Cluster_{}.csv'.format(cluster)

        d = pd.read_csv(filename)

        if scaler.lower() == 'standardize':
                sc = StandardScaler()

        elif scaler.lower() == 'minmax_zero':
                sc = MinMaxScaler(feature_range = (0, 1))

        elif scaler.lower() == 'minmax_one':
                sc = MinMaxScaler(feature_range = (-1, 1))

        elif scaler.lower() == 'normalize':
                sc = Normalizer(norm='l2')

        sc.fit(np.array(d['close']).reshape(len(d), 1))
        transformed = sc.inverse_transform(array)

        return transformed


def labelATR(row):
    if row['return_pred'] > row['ATR']:
        return 'up'
    elif row['return_pred'] < (-1) * row['ATR']:
        return 'down'
    else:
        return 'flat'

def label(row):
    if row['return_pred'] > 0:
        return 'up'
    elif row['return_pred'] < 0:
        return 'down'
    else:
        return 'flat'




def get_prediction_label(prediction, test, cluster, problem='Weekly_ATR'):
        filename = 'D:\\庫存健診開發\\data\\Training\\Raw\\Cluster_{}.csv'.format(cluster)

        data = pd.read_csv(filename, converters={'ts':str, 'StockNo': str})
        data['ts'] = pd.to_datetime(data['ts'])

        pred_list = prediction.tolist()
        for i, item in enumerate(test):
                pred_list[i].extend(test[i][:2])

        pred_df = pd.DataFrame(np.array(pred_list), columns=['prediction', 'ts', 'StockNo'])
        d = pd.merge(data, pred_df, on=['ts', 'StockNo'], how='left')

        df_list = [group[1] for group in d.groupby(d['StockNo'])]

        for df in df_list:
                if problem == 'Weekly_ATR':
                        df['return_pred'] = df['prediction'] - df['close'].shift(5)
                        df['label_pred'] = df.apply(labelATR, axis=1)

                elif problem == 'Weekly':
                        df['return_pred'] = df['prediction'] - df['close'].shift(5)
                        df['label_pred'] = df.apply(label, axis=1)

                elif problem == 'Daily_ATR':
                        df['return_pred'] = df['prediction'] - df['close'].shift(1)
                        df['label_pred'] = df.apply(labelATR, axis=1)
                
                elif problem == 'Daily':
                        df['return_pred'] = df['prediction'] - df['close'].shift(1)
                        df['label_pred'] = df.apply(label, axis=1)

                else:
                        return 'Problem not exist'

        
        d = pd.concat(df_list, axis=0)
        df_new = d[d['prediction'].notnull()].reset_index(drop=True)

        return df_new


def Regression_Evaluation(data, problem='Weekly_ATR'):
        
        if problem == 'Weekly_ATR':
                accuracy = len(data[data['label_weekly_ATR'] == data['label_pred']]) / len(data)

        elif problem == 'Weekly':
                accuracy = len(data[data['label_weekly'] == data['label_pred']]) / len(data)

        elif problem == 'Daily_ATR':
                accuracy = len(data[data['label_ATR'] == data['label_pred']]) / len(data)
                
        elif problem == 'Daily':
                accuracy = len(data[data['label'] == data['label_pred']]) / len(data)

        else:
                return 'Problem not exist'

        return accuracy



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
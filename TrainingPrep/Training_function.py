from tqdm import tqdm_notebook as tqdm
import calendar
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import math
import os
import datetime
from datetime import datetime, timedelta

def TrainTestSplit(data, TrainDate, ValDate):

    train_df = data[data['ts'].dt.date < TrainDate]
    val_df = data[data['ts'].dt.date < ValDate][data['ts'].dt.date >= TrainDate]
    test_df = data[data['ts'].dt.date >= ValDate]

    return [train_df, val_df, test_df]


def Normalization(data, scaler='standardize'):
    numerical = data[['12ema', '26ema', 'ATR', 'Intramonth', 
                        'MACD', 'RSI_15', 'SO%d3',
                        'SO%k5', 'TR', 'VWAP', 'WeekNum',
                        'close', 'high', 'low', 'lower_band', 'ma21', 'ma7', 'open',
                        'upper_band', 'vol']]

    df = data.copy()
    
    if scaler.lower() == 'standardize':
        sc = StandardScaler()

    elif scaler.lower() == 'minmax_zero':
        sc = MinMaxScaler(feature_range = (0, 1))

    elif scaler.lower() == 'minmax_one':
        sc = MinMaxScaler(feature_range = (-1, 1))

    elif scaler.lower() == 'normalize':
        sc = Normalizer(norm='l2')
    
    else:
        return 'No such scaler'

    a = sc.fit_transform(numerical)
    df[['12ema', '26ema', 'ATR', 'Intramonth', 
        'MACD', 'RSI_15', 'SO%d3',
        'SO%k5', 'TR', 'VWAP', 'WeekNum',
        'close', 'high', 'low', 'lower_band', 'ma21', 'ma7', 'open',
        'upper_band', 'vol']] = a[:]

    return df

##Form Training shape
#Regression 
def create_dataset_regression(dataset, lookback, forward):

    dataX = []
    dataY = []

    for i in range(0,dataset.shape[0]-lookback-forward,1):
        this_ds = dataset.iloc[i:i+lookback].copy()
        
        my_ds =  this_ds[['12ema', '26ema', 'ATR', 'Intramonth', 
                'MACD', 'RSI_15', 'SO%d3',
                'SO%k5', 'TR', 'VWAP', 'WeekNum',
                'close', 'high', 'low', 'lower_band', 'ma21', 'ma7', 'open',
                'upper_band', 'vol', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'#,
                #'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                #'First', 'Second', 'Third', 'Fourth', 'Fifth'
                ]]
                       
        dataX.append(np.array(my_ds))

        y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'close']]
            
        dataY.append(np.array(y))

    return [dataX, dataY]


def create_dataset_classification(dataset, lookback, forward, scaler='minmax_zero'):

    dataX = []
    dataY_ATR = []
    dataY = []

    for i in range(0,dataset.shape[0]-lookback-forward,1):
        this_ds = dataset.iloc[i:i+lookback].copy()

        numerical = this_ds[['12ema', '26ema', 'ATR', 'Intramonth', 
                        'MACD', 'RSI_15', 'SO%d3',
                        'SO%k5', 'TR', 'VWAP', 'WeekNum',
                        'close', 'high', 'low', 'lower_band', 'ma21', 'ma7', 'open',
                        'upper_band', 'vol']]

    
        if scaler.lower() == 'standardize':
            sc = StandardScaler()
        elif scaler.lower() == 'minmax_zero':
            sc = MinMaxScaler(feature_range = (0, 1))
        elif scaler.lower() == 'minmax_one':
            sc = MinMaxScaler(feature_range = (-1, 1))
        elif scaler.lower() == 'normalize':
            sc = Normalizer(norm='l2')
        else:
            return 'No such scaler'

        a = sc.fit_transform(numerical)
        
        this_ds[['12ema', '26ema', 'ATR', 'Intramonth', 
            'MACD', 'RSI_15', 'SO%d3',
            'SO%k5', 'TR', 'VWAP', 'WeekNum',
            'close', 'high', 'low', 'lower_band', 'ma21', 'ma7', 'open',
            'upper_band', 'vol']] = a[:]

        my_ds =  this_ds[['12ema', '26ema', 'ATR', 'Intramonth', 
                'MACD', 'RSI_15', 'SO%d3',
                'SO%k5', 'TR', 'VWAP', 'WeekNum',
                'close', 'high', 'low', 'lower_band', 'ma21', 'ma7', 'open',
                'upper_band', 'vol', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                'First', 'Second', 'Third', 'Fourth', 'Fifth']]
                       
        dataX.append(np.array(my_ds))

        if forward == 5:
            y = dataset['label_weekly'].iloc[i + lookback + forward - 1]
            y_ATR = dataset['label_weekly_ATR'].iloc[i + lookback + forward - 1]

        else:
            y = dataset['label'].iloc[i + lookback + forward - 1]
            y_ATR = dataset['label_ATR'].iloc[i + lookback + forward - 1]
            
        dataY.append(y)
        dataY_ATR.append(y_ATR)

    return [dataX, dataY, dataY_ATR]



def categorical_transform(data):

    onehot_encoder = OneHotEncoder(sparse=False)

    d = np.array(data).reshape(len(data), 1)
    Y = onehot_encoder.fit_transform(d)

    return Y
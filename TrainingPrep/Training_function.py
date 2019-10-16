from tqdm import tqdm_notebook as tqdm
import calendar
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
import os
import datetime
from datetime import datetime, timedelta

def TrainTestSplit(data, TrainDate):

<<<<<<< Updated upstream
    train_df = data[data['ts'].dt.date < TrainDate]
    val_df = data[data['ts'].dt.date < ValDate][data['ts'].dt.date >= TrainDate]
    test_df = data[data['ts'].dt.date >= ValDate]
=======
    train_df = data[data['ts'].dt.date < TrainDate][data['26ema'].notnull()]
    test_df = data[data['ts'].dt.date >= TrainDate][data['26ema'].notnull()]
>>>>>>> Stashed changes

    return train_df, test_df


def GetScalerParam(train_df, feature_list):
    df = pd.DataFrame(columns=['max', 'min', 'std', 'mean'])
    df['max'] = train_df[feature_list].max()
    df['min'] = train_df[feature_list].min()
    df['std'] = train_df[feature_list].std()
    df['mean'] = train_df[feature_list].mean()

    return df


def Scaler(data, feature_list, Param_df, problem='minmax0'):

    if problem.lower() not in ['standardize', 'minmax0']:
        return 'The scaler not exist'

    df = data.copy()

    if problem == 'standardize':
       std = Param_df['std']
       mean = Param_df['mean']

       for feature in feature_list:
           df[feature] = (df[feature] - mean.loc[feature])/std.loc[feature]

    elif problem == 'minmax0':
        Max = Param_df['max']
        Min = Param_df['min']

        for feature in feature_list:
            df[feature] = (df[feature] - Min.loc[feature])/(Max.loc[feature] - Min.loc[feature])

    return df
    



##Form Training shape
#Regression 
def create_dataset(dataset, lookback, forward, problem='regression'):

    if problem.lower() not in ['regression', 'classification']:
        return 'Problem not exit'

    dataX = []
    dataY = []

    if problem == 'classification':
        dataY_ATR = []

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

        if problem == 'regression':
            y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'origin_close']]
            dataY.append(np.array(y))

        else:
            if forward == 5:
                y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'label_weekly']]
                y_ATR = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'label_weekly_ATR']]

            else:
                y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'label']]
                y_ATR = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'label_ATR']] 
            
            dataY.append(np.array(y))
            dataY_ATR.append(np.array(y_ATR))

    if problem == 'regression':
        return [dataX, dataY]

    else:
        return[dataX, dataY, dataY_ATR]






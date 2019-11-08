from tqdm import tqdm_notebook as tqdm
import calendar
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
import os
import datetime
from datetime import datetime, timedelta, date

def TrainTestSplit(data, TrainDate, valDate):

    train_df = data[data['ts'].dt.date < TrainDate][data['ema26'].notnull()]
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    val_df = data[data['ts'].dt.date < valDate][data['ts'].dt.date >= TrainDate][data['ema26'].notnull()]
    val_df = val_df.replace([np.inf, -np.inf], np.nan)
    test_df = data[data['ts'].dt.date >= valDate][data['ema26'].notnull()]
    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    train_df, val_df, test_df = train_df.fillna(0), val_df.fillna(0), test_df.fillna(0)

    return train_df, val_df, test_df


def ClusterSplit(cluster_train, cluster_test, data, filepath, train_date=date(2017,9,1)):

    '''
    Split data into different clusters.
    Input the dataframe containing labels(clusterdata), dataframe containing all stocks, total cluster and writing path
    '''

    train = data[data['ts'].dt.date < train_date]
    test = data[data['ts'].dt.date >= train_date]
    cluster_num = len(cluster_train['cluster'].unique())

    for i in tqdm(range(cluster_num)):
        stock_list_train = [str(i) for i in cluster_train[cluster_train['cluster'] == i].index.tolist()]
        stock_list_test = [str(i) for i in cluster_test[cluster_test['cluster'] == i].index.tolist()]
        train_df = train[train['StockNo'].isin(stock_list_train)]
        test_df = test[test['StockNo'].isin(stock_list_test)]

        df = pd.concat([train_df, test_df], axis=0)
        df.to_csv(filepath + 'Cluster_{}.csv'.format(i), index=False)


def total_split(cluster_train, cluster_test, data, filepath, train_date=date(2017,9,1)):

    '''
    Split data into different clusters.
    Input the dataframe containing labels(clusterdata), dataframe containing all stocks, total cluster and writing path
    '''

    train = data[data['ts'].dt.date < train_date]
    test = data[data['ts'].dt.date >= train_date]
    train_list, test_list = [], []

    for i in tqdm(range(5)):
        stock_list_train = [str(i) for i in cluster_train[cluster_train['cluster'] == i].index.tolist()]
        stock_list_test = [str(i) for i in cluster_test[cluster_test['cluster'] == i].index.tolist()]
        train_df = train[train['StockNo'].isin(stock_list_train)]
        train_df['cluster'] = i
        test_df = test[test['StockNo'].isin(stock_list_test)]
        test_df['cluster'] = i

        train_list.append(train_df)
        test_list.append(test_df)

    train, test = pd.concat(train_list, axis=0), pd.concat(test_list, axis=0)
    df = pd.concat([train, test], axis=0)
    df.to_csv(filepath + 'large.csv', index=False)


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
def create_dataset(dataset, lookback, forward, feature_list, problem='classification'):

    if problem.lower() not in ['regression', 'classification']:
        return 'Problem not exit'

    dataX = []
    dataY = []

    dataset = dataset.sort_values(by='ts').reset_index(drop=True)
    for i in range(0,dataset.shape[0]-lookback-forward,1):
        this_ds = dataset.iloc[i:i+lookback].copy()
        
        my_ds =  this_ds[feature_list]
                       
        dataX.append(np.array(my_ds))

        if problem == 'regression':
            y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'origin_close']]
            dataY.append(np.array(y))

        else:
            price_return =  dataset.iloc[i + lookback + forward - 1]['origin_close'] - dataset.iloc[i + lookback - 1]['origin_close']
            if price_return > 0:
                y = 1
            else:
                y = 0

            Y =  dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo']].tolist()
            Y.append(y)
            dataY.append(Y)

    return [dataX, dataY]


def separate_elimination(data, lookback, forward, feature_list, problem='classification'):
    d = data.copy()

    if len(d[d['eliminate'] != 0]) == 0:
        return create_dataset(d, lookback, forward, feature_list, problem='classification')

    start_date = d[d['eliminate'] == 1]['ts'].tolist()
    start_date.sort(reverse=True)
    dataX = []
    dataY = []
    for start in start_date:
        d1 = d[d['ts'] <= start]
        result = create_dataset(d1, lookback, forward, feature_list, problem='classification')
        dataX.extend(result[0])
        dataY.extend(result[1])

        d = d[d['ts'] > start]

    result = create_dataset(d, lookback, forward, feature_list, problem='classification')
    dataX.extend(result[0])
    dataY.extend(result[1])

    return [dataX, dataY]


def transform_whole_csv(X, Y, feature_list):

    X = np.array(X)
    X_new = X.reshape(-1, X.shape[1] * X.shape[2])

    column_list = [f'{feature}_day{i+1}' for i in range(20) for feature in feature_list]

    df = pd.DataFrame(data=X_new[0:,0:], index=[i for i in range(X_new.shape[0])], columns=column_list)
    df['Y'] = [item[2] for item in Y]
    df['ts'] = [item[0] for item in Y]
    df['StockNo'] = [item[1] for item in Y]

    return df


def transform_cluster_csv(X, Y, cluster_num, feature_list):

    X = np.array(X)
    X_new = X.reshape(-1, X.shape[1] * X.shape[2])

    column_list = [f'{feature}_day{i+1}' for i in range(20) for feature in feature_list]

    df = pd.DataFrame(data=X_new[0:,0:], index=[i for i in range(X_new.shape[0])], columns=column_list)
    df['Y'] = [item[2] for item in Y]
    df['ts'] = [item[0] for item in Y]
    df['StockNo'] = [item[1] for item in Y]
    df['cluster'] = str(cluster_num)

    return df
    
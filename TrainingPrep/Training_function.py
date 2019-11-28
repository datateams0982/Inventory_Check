from tqdm import tqdm_notebook as tqdm
import numpy as np  
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
import math
import os
import datetime
from datetime import datetime, timedelta, date

def TrainTestSplit(data, TrainDate, valDate):

    train_df = data[data['ts'].dt.date < TrainDate][data['pvt'].notnull()][data['MACD_diff'].notnull()]
    val_df = data[data['ts'].dt.date < valDate][data['ts'].dt.date >= TrainDate][data['pvt'].notnull()][data['MACD_diff'].notnull()]
    test_df = data[data['ts'].dt.date >= valDate][data['pvt'].notnull()][data['MACD_diff'].notnull()]

    return train_df, val_df, test_df


# def ClusterSplit(cluster_train, cluster_test, data, filepath, train_date=date(2017,9,1)):

#     '''
#     Split data into different clusters.
#     Input the dataframe containing labels(clusterdata), dataframe containing all stocks, total cluster and writing path
#     '''

#     train = data[data['ts'].dt.date < train_date]
#     test = data[data['ts'].dt.date >= train_date]
#     cluster_num = len(cluster_train['cluster'].unique())

#     for i in tqdm(range(cluster_num)):
#         stock_list_train = [str(i) for i in cluster_train[cluster_train['cluster'] == i].index.tolist()]
#         stock_list_test = [str(i) for i in cluster_test[cluster_test['cluster'] == i].index.tolist()]
#         train_df = train[train['StockNo'].isin(stock_list_train)]
#         test_df = test[test['StockNo'].isin(stock_list_test)]

#         df = pd.concat([train_df, test_df], axis=0)
#         df.to_csv(filepath + 'Cluster_{}.csv'.format(i), index=False)


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

    return df


def GetScalerParam(train_df, feature_list, industry_index):
    df = pd.DataFrame(columns=['max', 'min', 'std', 'mean'])
    df['max'] = train_df[feature_list].max()
    df['min'] = train_df[feature_list].min()
    df['std'] = train_df[feature_list].std()
    df['mean'] = train_df[feature_list].mean()

    industry_list = train_df['industry'].unique().tolist()
    reference = {industry: pd.DataFrame(columns=['max', 'min', 'std', 'mean']) for industry in industry_list}

    for industry in industry_list:
        d = train_df[train_df['industry'] == industry]
        reference[industry]['max'] = d[industry_index].max()
        reference[industry]['min'] = d[industry_index].min()
        reference[industry]['std'] = d[industry_index].std()
        reference[industry]['mean'] = d[industry_index].mean()

    return df, reference



def Scaler(data, Param_df, industry_reference, feature_list, industry_index, problem='minmax0'):

    assert problem in ['standardize', 'minmax0']

    industry_list = data['industry'].unique().tolist()
    df_list = []
    data['origin_VWAP'] = data['VWAP']
    data['origin_ATR'] = data['ATR']

    for industry in industry_list:
        industry_param = industry_reference[industry]
        df = data[data.industry == industry]

        if problem == 'standardize':
            std = Param_df['std']
            mean = Param_df['mean']
            industry_std = industry_param['std']
            industry_mean = industry['mean']

            for col in industry_index:
                df[f'{col}_scaled'] = (df[col] - industry_mean.loc[col])/industry_std.loc[col]

            for feature in feature_list:
                df[feature] = (df[feature] - mean.loc[feature])/std.loc[feature]
        

        elif problem == 'minmax0':
            Max = Param_df['max']
            Min = Param_df['min']
            industry_max = industry_param['max']
            industry_min = industry_param['min']

            for col in industry_index:
                df[f'{col}_scaled'] = (df[col] - industry_min.loc[col])/(industry_max.loc[col] - industry_min.loc[col])

            for feature in feature_list:
                df[feature] = (df[feature] - Min.loc[feature])/(Max.loc[feature] - Min.loc[feature])
        
        df_list.append(df)

    
    df = pd.concat(df_list, axis=0)

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
            y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'VWAP']]
            dataY.append(np.array(y))

        else:
            price_return =  dataset.iloc[i + lookback + forward - 1]['origin_VWAP'] - dataset.iloc[i + lookback - 1]['origin_VWAP']
            # if price_return > 0:
            #     y = 1
            # else:
            #     y = 0
            if price_return > dataset.iloc[i + lookback - 1]['origin_ATR']:
                y = 1
            elif price_return < (-1) * dataset.iloc[i + lookback - 1]['origin_ATR']:
                y = -1
            else:
                y = 0

            Y =  dataset.iloc[i + lookback - 1][['ts', 'StockNo', 'cluster', 'industry']].tolist()
            Y.append(y)
            dataY.append(Y)

    return [dataX, dataY]


def separate_elimination(data, lookback, forward, feature_list, problem='classification'):
    d = data.copy()

    if len(d[d['eliminate'] != 0]) == 0:
        return create_dataset(d, lookback, forward, feature_list, problem='classification')

    start_date = d[d['eliminate'] == 2]['ts'].tolist()
    start_date.sort(reverse=True)
    dataX = []
    dataY = []
    for start in start_date:
        d1 = d[d['ts'] < start]
        result = create_dataset(d1, lookback, forward, feature_list, problem='classification')
        dataX.extend(result[0])
        dataY.extend(result[1])

        d = d[d['ts'] >= start]

    result = create_dataset(d, lookback, forward, feature_list, problem='classification')
    dataX.extend(result[0])
    dataY.extend(result[1])

    return [dataX, dataY]

def reemovNestings(l): 
    output = [] 
    for i in l: 
        if type(i) == list: 
            reemovNestings(i) 
        else: 
            output.append(i) 

    

def transform_whole_csv(X, Y, feature_list, lookback=20):

    X = np.array(X)
    X_new = X.reshape(-1, X.shape[1] * X.shape[2])

    column_list = [f'{feature}_day{i+1}' for i in range(lookback) for feature in feature_list]

    df = pd.DataFrame(data=X_new[0:,0:], index=[i for i in range(X_new.shape[0])], columns=column_list)
    df['Y'] = [item[4] for item in Y]
    df['ts'] = [item[0] for item in Y]
    df['StockNo'] = [item[1] for item in Y]
    df['cluster'] = [item[2] for item in Y]
    df['industry'] = [item[3] for item in Y]

    return df


# def transform_cluster_csv(X, Y, cluster_num, feature_list, lookback=20):

#     X = np.array(X)
#     X_new = X.reshape(-1, X.shape[1] * X.shape[2])

#     column_list = [f'{feature}_day{i+1}' for i in range(lookback) for feature in feature_list]

#     df = pd.DataFrame(data=X_new[0:,0:], index=[i for i in range(X_new.shape[0])], columns=column_list)
#     df['Y'] = [item[2] for item in Y]
#     df['ts'] = [item[0] for item in Y]
#     df['StockNo'] = [item[1] for item in Y]
#     df['cluster'] = str(cluster_num)

#     return df
    
import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import random
import os
import pickle
from sklearn import metrics
from datetime import date
from sklearn.metrics import classification_report
import json

def predict_up(row, threshold):
    
    if row['Y_1_score'] > threshold:
        return 1
    else:
        return 0


def predict_down(row, threshold):
    
    if row['Y_0_score'] > threshold:
        return 0
    else:
        return 1


def Evaluation_up(data, threshold=0.5):
    
    data['prediction'] = data.apply(predict_up, threshold=threshold, axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(data['Y'], data['prediction'])
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(data['Y'], data['prediction'])

    target_names = ['down', 'up']
    report = classification_report(data['Y'].tolist(), data['prediction'].tolist(), target_names=target_names)

    return [auc, accuracy, report]


def Evaluation_down(data, threshold=0.5):
    
    data['prediction'] = data.apply(predict_down, threshold=threshold, axis=1)
    fpr, tpr, thresholds = metrics.roc_curve(data['Y'], data['prediction'])
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(data['Y'], data['prediction'])

    target_names = ['down', 'up']
    report = classification_report(data['Y'].tolist(), data['prediction'].tolist(), target_names=target_names)

    return [auc, accuracy, report]


def Separate_Evaluation(data, threshold=0.5, prediction=False, **kwargs):
    
    if prediction:
        data['prediction'] = data.apply(predict_up, threshold=threshold, axis=1)
        data['ts'] = pd.to_datetime(data['ts'])
        
    else:
        data = data

    if (len(kwargs.keys()) == 1):
        if 'year' in kwargs:
            year = kwargs['year']
            df = data[data.ts.dt.year == year]
        else:
            stock = kwargs['stock']
            df = data[data.StockNo == stock]

    target_names = ['down', 'up']

    fpr, tpr, thresholds = metrics.roc_curve(df['Y'], df['prediction'])
    auc = metrics.auc(fpr, tpr)

    report = classification_report(df['Y'], df['prediction'], target_names=target_names)
    accuracy = metrics.accuracy_score(df['Y'], df['prediction'])

    return report, accuracy, auc


def Overall_Evaluation(data, threshold=0.6):
    
    data['prediction'] = data.apply(predict_up, threshold=0.5, axis=1)
    data['ts'] = pd.to_datetime(data['ts'])
    df_list = [group[1] for group in data.groupby(data['StockNo'])]
    
    stock_list = []
    cluster_list = []
    acc_list = []
    for i, d in enumerate(tqdm(df_list)):
        acc = metrics.accuracy_score(d['Y'], d['prediction'])
        acc_list.append(acc)
        if  acc< threshold:
            stock_list.append(d['StockNo'].iloc[0])
            cluster_list.append(d['cluster'].iloc[0])
            
    rate_acc = len(stock_list)/len(df_list)



    return rate_acc, stock_list, cluster_list, acc_list


def read_json(file_path):

    d = []
    with open(file_path , 'r') as fp:
        for line in fp:
            d.append(json.loads(line))

    Y = [int(item['Y']) for item in d]
    ts = [item['ts'] for item in d]
    StockNo = [item['StockNo'] for item in d]
    cluster = [item['cluster'] for item in d]
    Y_down = [float(item['predicted_Y'][0]['tables']['score']) for item in d]
    Y_up = [float(item['predicted_Y'][1]['tables']['score']) for item in d]

    df = pd.DataFrame(np.stack([Y, ts, StockNo, cluster, Y_down, Y_up], axis=1), columns=['Y', 'ts', 'StockNo', 'cluster', 'Y_0_score', 'Y_1_score'])

    df['Y_1_score'] = df['Y_1_score'].astype(np.float64)
    df['Y_0_score'] = df['Y_0_score'].astype(np.float64)
    df['Y'] = df['Y'].astype(np.int)

    return df
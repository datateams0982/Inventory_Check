import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import os
import pickle
from sklearn import metrics
from datetime import date, timedelta, datetime
import time
from sklearn.metrics import classification_report
import json
import pymssql as mssql
import calendar
import matplotlib.pyplot as plt


def read_prediction(file_path, origin_path):

    file_list = os.listdir(file_path)
    df_list = []

    for i, filename in enumerate(tqdm(file_list)):
        if filename[-3:] != 'csv':
            continue

        df = pd.read_csv(file_path+filename, converters={'StockNo': str, 'ts': str})
        df_list.append(df)

    prediction = pd.concat(df_list, axis=0)
    prediction['ts'] = pd.to_datetime(prediction['ts'])
    original = pd.read_csv(origin_path, converters={'StockNo': str, 'ts': str}, usecols=['ts', 'StockNo', 'Y', 'total'])
    original['ts'] = pd.to_datetime(original['ts'])

    combine = pd.merge(original, prediction[['ts', 'StockNo', 'Y_0_score', 'Y_1_score']], on=['ts', 'StockNo'], how='inner')
    combine['ts'] = pd.to_datetime(combine['ts'])

    return prediction, combine


def predict(row, threshold):
    
    if row['Y_1_score'] > threshold:
        return 1
    elif row['Y_0_score'] > threshold:
        return 0
    else:
        return -1



def Evaluation(data, threshold=0.5):
    
    d = data.copy()
    d['prediction'] = d.apply(predict, threshold=threshold, axis=1)
    d = d[d.prediction != -1]
    accuracy = metrics.accuracy_score(d['Y'], d['prediction'])

    target_names = ['down', 'up']
    report = classification_report(d['Y'].tolist(), d['prediction'].tolist(), target_names=target_names)
    up_support = len(data[data.Y_1_score > threshold])
    down_support = len(data[data.Y_0_score > threshold])
    up_support_ratio = len(data[data.Y_1_score > threshold]) / len(data)
    down_support_ratio = len(data[data.Y_0_score > threshold]) / len(data)

    return [accuracy, report, up_support, down_support, up_support_ratio, down_support_ratio]



def Separate_Evaluation(data, threshold=0.5, prediction=False, **kwargs):
    
    data['ts'] = pd.to_datetime(data['ts'])
    d = data.copy()
    if prediction:
        d['prediction'] = d.apply(predict, threshold=threshold, axis=1)
        d = d[d.prediction != -1]
        d['ts'] = pd.to_datetime(d['ts'])
        
    else:
        d = d

    if (len(kwargs.keys()) == 1):
        if 'year' in kwargs:
            year = kwargs['year']
            df = d[d.ts.dt.year == year]
        else:
            stock = kwargs['stock']
            df = d[d.StockNo == stock]

    if (len(kwargs.keys()) == 2):

        if ('year' in kwargs) and ('month' in kwargs) :
            year = kwargs['year']
            month = kwargs['month']
            df = d[(d.ts.dt.year == year) & (d.ts.dt.month == month)]

    target_names = ['down', 'up']

    report = classification_report(df['Y'], df['prediction'], target_names=target_names)
    accuracy = metrics.accuracy_score(df['Y'], df['prediction'])
    up_support = len(df[df.Y_1_score > threshold])
    down_support = len(df[df.Y_0_score > threshold])
    up_support_ratio = len(df[df.Y_1_score > threshold]) / len(data[data.ts.dt.year == year])
    down_support_ratio = len(df[df.Y_0_score > threshold]) / len(data[data.ts.dt.year == year])

    return [accuracy, report, up_support, down_support, up_support_ratio, down_support_ratio]



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

    df['ts'] = df['ts'].apply(lambda x: str(x)[:10])
    df['Y_1_score'] = df['Y_1_score'].astype(np.float64)
    df['Y_0_score'] = df['Y_0_score'].astype(np.float64)
    df['Y'] = df['Y'].astype(np.int)
    # df['cluster'] = df['cluster'].astype(np.int)

    return df


def check(row):
    if (row['Y_1_score'] > 0.5) and (row['Y'] == 1):
        return True
    elif (row['Y_1_score'] <= 0.5) and (row['Y'] == 0):
        return True
    else:
        return False


def plot_movement(price_data, prediction):

    d = pd.merge(price_data, prediction,  on=['ts', 'StockNo'], how='inner')
    d['check'] = d.apply(check, axis=1)
    df_list = [group[1] for group in d.groupby(d['StockNo'])]

    for df in df_list:
        plt.figure(figsize=(10,10))
        plt.plot(d['future_return']/d['future_return'].max())
        plt.plot(d['Y_1_score'] - 0.5)


def read_original_data(path):
    
    data = pd.read_csv(path, converters={'ts': str, 'StockNo': str, 'StockName': str}, usecols=["ts", "StockNo", 'close', 'close_return', "VWAP_day5", 'VWAP_after', 'index_close', 'foreign_ratio', 'investment_ratio', 'corporation_ratio'])
    df_list = [group[1] for group in data.groupby(data['StockNo'])]
    output_list = []

    for df in df_list:
        df = df.sort_values(by='ts').reset_index(drop=True)
        df['future_return'] = df['VWAP_after'] - df['VWAP_day5']
        output_list.append(df)
        
    df = pd.concat(output_list, axis=0)
    df['ts'] = pd.to_datetime(df['ts'])
    
    return df


def verify(row):
    if (row['future_return'] > 0) and (row['Y'] == 1):
        return True
    elif (row['future_return'] <= 0) and (row['Y'] == 0):
        return True
    else:
        return False


def verification(original_data, label_data):

    original_data['ts'], label_data['ts'] = pd.to_datetime(original_data['ts']), pd.to_datetime(label_data['ts'])
    d = pd.merge(original_data, label_data, on=['ts', 'StockNo'], how='inner')
    d = d[d.future_return.notnull()]
    d['verify'] = d.apply(verify, axis=1)

    if len(d[~d.verify]) == 0:
        return True
    else:
        return False


def send_query(query, db = 'OpenData', timecost = True, showcol = True, showlen = True):
    
    tStart = time.time()
    ods = mssql.connect(host = '128.110.13.89', 
                      user = '011553', 
                      password = 'Sino821031pac', 
                      charset='utf8')

    odscur = ods.cursor(as_dict = True)
    odscur.execute(query)
    temp = odscur.fetchall()
    df = pd.DataFrame(temp)
    odscur.close()
    tEnd = time.time()

    if timecost == True:
        print("It cost %f sec" % (tEnd - tStart))
    if showlen == True:
        print('Data length:', len(df))
    if showcol == True:
        print(df.columns)

    return df

def get_daily_index(start_date, end_date):
    
    start_date = start_date.strftime('%Y%m%d')
    end_date = end_date.strftime('%Y%m%d')
    index_subquery = f'''SELECT [DATE] AS ts,
                                TRY_CAST([OPEN] AS FLOAT) AS [index_open], 
                                TRY_CAST([HIGH] AS FLOAT) AS [index_high],
                                TRY_CAST([LOW] AS FLOAT) AS [index_low],
                                TRY_CAST([CLOSE] AS FLOAT) AS [index_close],
                                TRY_CAST([VOLUME] AS FLOAT) AS index_vol
                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                        WHERE 
                            STOCK_ID = 'TWA00' 
                            AND DATE BETWEEN {start_date} AND {end_date}
                        '''
    
    index_df = send_query(index_subquery, timecost = False, showcol = False, showlen = False)
    index_df['ts'] = index_df['ts'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    index_df['ts'] = pd.to_datetime(index_df['ts'])
    
    return index_df



def find_consecutive(data, column, criteria=0.5, direction='low'):

    max_length = 0
    start_point = 0
    end_point = 0
    max_end_point = 0

    series = data[column]

    length_list = []
    end_point_list = []
    

    if direction == 'low':
        for i in range(len(series)):
            if series.iloc[i] <= criteria:
                continue
            else:
                l = (i - start_point)
                start_point = i + 1
                end_point = i - 1
                length_list.append(l)
                end_point_list.append(series.index[end_point])
                if l > max_length:
                    max_length = l 
                    max_end_point = end_point       
                else:
                    continue

    else:
        for i in range(len(series)):
            if series.iloc[i] > criteria:
                continue
            else:
                l = (i - start_point)
                start_point = i + 1
                end_point = i - 1
                length_list.append(l)
                end_point_list.append(series.index[end_point])
                if l > max_length:
                    max_length = l 
                    max_end_point = end_point       
                else:
                    continue

    max_point = series.index[max_end_point]
    return_list = series.iloc[(max_end_point - max_length + 1):max_end_point]

    return [length_list, end_point_list, max_point, return_list]


def daily_evaluation(price_data, prediction, threshold=0.5):

    price_data['ts'], prediction['ts'] = pd.to_datetime(price_data['ts']), pd.to_datetime(prediction['ts'])
    d = pd.merge(price_data, prediction,  on=['ts', 'StockNo'], how='inner')
    d = d[d.future_return.notnull()]
    d['prediction'] = d.apply(predict, threshold=threshold, axis=1)
    d = d[d.prediction != -1]
    
    index_list = d['ts'].dt.date.unique().tolist()
    df = pd.DataFrame(columns=['accuracy', 'return', 'ratio', 'portfolio', 'up_precision', 'down_precision', 'up_support', 'down_support', 'index'], index=index_list)

    for date_string in index_list:
        d1 = d[d.ts.dt.date == date_string]
        df.loc[date_string, 'accuracy'] = metrics.accuracy_score(d1['Y'], d1['prediction'])
        df.loc[date_string, 'index'] = d1['index_close'].iloc[0]
        up = d1[d1.prediction == 1]
        down = d1[d1.prediction == 0]
        df.loc[date_string, 'return'] = up['future_return'].mean()
        df.loc[date_string, 'ratio'] = (up['future_return'] / up['VWAP_day5']).mean()
        df.loc[date_string, 'portfolio'] = up['future_return'].sum() / up['VWAP_day5'].sum()
        df.loc[date_string, 'up_support'] = len(up)
        df.loc[date_string, 'down_support'] = len(down)
        df.loc[date_string, 'up_precision'] = len(up[up.Y == 1])/len(up)
        df.loc[date_string, 'down_precision'] = len(down[down.Y == 0])/len(down)

    return df


def Stock_evaluation(price_data, prediction, threshold=0.5):

    price_data['ts'], prediction['ts'] = pd.to_datetime(price_data['ts']), pd.to_datetime(prediction['ts'])
    d = pd.merge(price_data, prediction,  on=['ts', 'StockNo'], how='inner')
    d = d[d.future_return.notnull()]

    d['prediction'] = d.apply(predict, threshold=threshold, axis=1)
    d = d[d.prediction != -1]
    index_list = d['StockNo'].unique().tolist()
    df = pd.DataFrame(columns=['accuracy', 'return', 'ratio', 'portfolio', 'up_precision', 'down_precision', 'up_support', 'down_support', 'foreign_ratio', 'investment_ratio', 'corporation_ratio'], index=index_list)

    for stock in index_list:
        d1 = d[d.StockNo == stock]
        df.loc[stock, 'accuracy'] = metrics.accuracy_score(d1['Y'], d1['prediction'])
        df.loc[stock, 'foreign_ratio'] = d1['foreign_ratio'].mean()
        df.loc[stock, 'investment_ratio'] = d1['investment_ratio'].mean()
        df.loc[stock, 'corporation_ratio'] = d1['corporation_ratio'].mean()
        up = d1[d1.prediction == 1]
        down = d1[d1.prediction == 0]
        df.loc[stock, 'return'] = up['future_return'].mean()
        df.loc[stock, 'ratio'] = (up['future_return'] / up['VWAP_day5']).mean()
        df.loc[stock, 'portfolio'] = up['future_return'].sum() / up['VWAP_day5'].sum()
        
        df.loc[stock, 'up_support'] = len(up)
        df.loc[stock, 'up_support'] = len(up)
        df.loc[stock, 'down_support'] = len(down)
        if len(up) == 0:
            df.loc[stock, 'up_precision'] = 0
        else:
            df.loc[stock, 'up_precision'] = len(up[up.Y == 1])/len(up)
        if len(down) == 0:
            df.loc[stock, 'down_precision'] = 0
        else:    
            df.loc[stock, 'down_precision'] = len(down[down.Y == 0])/len(down)

    return df


def monthly_evaluation(price_data, prediction, threshold=0.5):

    price_data['ts'], prediction['ts'] = pd.to_datetime(price_data['ts']), pd.to_datetime(prediction['ts'])
    d = pd.merge(price_data, prediction,  on=['ts', 'StockNo'], how='inner')
    d = d[d.future_return.notnull()]

    d['prediction'] = d.apply(predict, threshold=threshold, axis=1)
    d = d[d.prediction != -1]
    year_list = d['ts'].dt.year.unique().tolist()
    month_list = d['ts'].dt.month.unique().tolist()

    year_month_list = []

    for y in year_list:
        for m in month_list:
            d1 = d[(d.ts.dt.year == y) & (d.ts.dt.month == m)]
            if len(d1) > 100:
                year_month_list.append([y, m])
            else:
                continue

    index_list = [date(item[0], item[1], 1) for item in year_month_list]
    df = pd.DataFrame(columns=['accuracy', 'up_return', 'up_ratio', 'up_portfolio', 'down_return', 'down_ratio', 'down_portfolio', 'up_precision', 'down_precision', 'up_support_mean', 'down_support_mean', 'index'], index=index_list)

    for m in year_month_list:
        d1 = d[(d.ts.dt.year == m[0]) & (d.ts.dt.month == m[1])]
        d1 = d1.sort_values(by='ts')
        date_string = date(m[0], m[1], 1)
        df.loc[date_string, 'accuracy'] = metrics.accuracy_score(d1['Y'], d1['prediction'])
        df.loc[date_string, 'index'] = d1['index_close'].iloc[-1]
        up = d1[d1.prediction == 1]
        down = d1[d1.prediction == 0]
        df.loc[date_string, 'up_return'] = up['future_return'].mean()
        df.loc[date_string, 'up_ratio'] = (up['future_return'] / up['VWAP_day5']).mean()
        df.loc[date_string, 'up_portfolio'] = up['future_return'].sum() / up['VWAP_day5'].sum()
        df.loc[date_string, 'down_return'] = (-1) * down['future_return'].mean()
        df.loc[date_string, 'down_ratio'] = (-1) * (down['future_return'] / down['VWAP_day5']).mean()
        df.loc[date_string, 'down_portfolio'] = (-1) * down['future_return'].sum() / down['VWAP_day5'].sum()
        df.loc[date_string, 'up_support_mean'] = len(up)/len(d1.ts.unique())
        df.loc[date_string, 'down_support_mean'] = len(down)/len(d1.ts.unique())
        df.loc[date_string, 'up_precision'] = len(up[up.Y == 1])/len(up)
        df.loc[date_string, 'down_precision'] = len(down[down.Y == 0])/len(down)


    return df



def daily_aggregate(daily_eval_df, **kwargs):

    daily_eval_df['ts'] = daily_eval_df.index
    if 'year' in kwargs:
        y = kwargs['year']
        d = daily_eval_df[daily_eval_df.ts.dt.year == y]
    else:
        d = daily_eval_df.copy()

    index_list = ['accuracy', 'return', 'ratio', 'portfolio', 'up_precision', 'down_precision', 'up_support', 'down_support']
    df = pd.DataFrame(columns=['min', 'max', 'std', 'mean', 'min_date', 'max_date', 'worst', 'worst_date', 'worst_mean', 'best', 'best_date', 'best_mean'], index=index_list)
    
    acc_threshold = d['accuracy'].mean() - d['accuracy'].std()
    up_support_threshold = d['up_support'].mean() - d['up_support'].std()
    down_support_threshold = d['down_support'].mean() - d['down_support'].std()
    up_precision_threshold = d['up_precision'].mean() - d['up_precision'].std()
    down_precision_threshold = d['down_precision'].mean() - d['down_precision'].std()
    threshold_list = [acc_threshold, 0, 0, 0, up_support_threshold, down_support_threshold, up_precision_threshold, down_precision_threshold]

    return_dict = {}

    for index, t in zip(index_list, threshold_list):
        df.loc[index, 'min'] = d[index].min()
        df.loc[index, 'max'] = d[index].max()
        df.loc[index, 'std'] = d[index].std()
        df.loc[index, 'mean'] = d[index].mean()
        df.loc[index, 'min_date'] = d[d[index] == d[index].min()].index[0]
        df.loc[index, 'max_date'] = d[d[index] == d[index].max()].index[0]
        worse_r = find_consecutive(d, index, criteria=t, direction='low')
        best_r = find_consecutive(d, index, criteria=t, direction='high')

        df.loc[index, 'worst'] = max(worse_r[0])
        df.loc[index, 'best'] = max(best_r[0])
        df.loc[index, 'worst_date'] = worse_r[2]
        df.loc[index, 'best_date'] = best_r[2]
        df.loc[index, 'worst_mean'] = np.array(worse_r[0]).mean()
        df.loc[index, 'best_mean'] = np.array(worse_r[0]).mean()
        return_dict[f'{index}_worst'] = worse_r[3]
        return_dict[f'{index}_best'] = best_r[3]

    return df, return_dict


def plot_precision_daily(df, figure_size):

    d = df.sort_values
    d['index_close_normalize'] = (d['index'] - d['index'].min()) / (d['index'].max() - d['index'].min())

    fig = plt.figure(figsize=figure_size)
    ax = plt.subplot2grid((4,2), (0,1))
    ax2 = ax.twinx()
    plt.plot(d['ts'], d['up_precision'], ax=ax)
    plt.plot(d['ts'], d['down_precision'], ax=ax)
    plt.plot(d['ts'], d['index_close_normalize'], ax=ax2)
    fig.show()



    
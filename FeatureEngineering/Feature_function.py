import numpy as np 
import pandas as pd 
import datetime
from datetime import datetime, timedelta, date
from scipy.stats import skew, kurtosis
import math
import pywt 
import copy


def TR(row):
    TR = max([(row["high"] - row["low"]), abs(row["high"] - row["close_lag"]), abs(row["close_lag"] - row["low"])])
    
    return TR

#Compute RSI
def RSI(df, n):  
    df = df.reset_index(drop=True)
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')  
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(UpI.ewm(span = n, min_periods = n - 1).mean())  
    NegDI = pd.Series(DoI.ewm(span = n, min_periods = n - 1).mean())  

    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n) + '_week')  
    RSI = RSI.replace([np.inf, -np.inf], np.nan)
    RSI = RSI.fillna(0) 
    df = df.join(RSI) 

    return df


#Compute Stochastic Index
def STO(df, nk, nD):  
    df = df.reset_index(drop=True)
    SOk = pd.Series((df['close'] - df['low'].rolling(nk).min()) / (df['high'].rolling(nk).max() - df['low'].rolling(nk).min()), name = 'SOk'+str(nk)+'_week') 
    SOk = SOk.replace([np.inf, -np.inf], np.nan)
    SOk = SOk.fillna(0)
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SOd'+str(nD)+'_week')
    SOd = SOd.fillna(0)
    df = df.join(SOk)
    df = df.join(SOd)
    
    return df


def price_volume_trend(data):

    d = data.sort_values(by='ts').reset_index(drop=True)
    d['pvt_current'] = d['vol'] * (d['close'] - d['close_lag']) / d['close_lag']

    for item in ['index_', 'industry_']:
        d[f'{item}pvt_current'] = d[f'{item}vol'] * (d[f'{item}close'] - d[f'{item}close'].shift(1)) / d[f'{item}close'].shift(1)


    feature = [f'{item}pvt_current' for item in ['index_', 'industry_', '']]
    pvt_feature = [f'{item}pvt_week' for item in ['index_', 'industry_', '']]
    
    for f in pvt_feature:
        d[f] = 0

    d.loc[:, pvt_feature] = d.loc[:, feature].rolling(window=20).sum().values

    d = d.drop(columns=feature)

    return d



def index_slope(row, problem='close'):

    if problem == 'close':
        slope = row['return'] - row['index_return']
    else:
        slope = row['VWAP_return'] - row['index_return']

    return slope

def industry_slope(row, problem='close'):

    if problem == 'close':
        slope = row['return'] - row['industry_return']
    else:
        slope = row['VWAP_return'] - row['industry_return']

    return slope


def label(row):

    if row['VWAP_after'] - row['close'] > 0:
        return 1
    elif row['VWAP_after'] - row['close'] <= 0:
        return 0


def log_return(row, problem='close'):

    assert problem in ['close', 'VWAP', 'index', 'industry']

    if problem == 'close':
        result = ((row['close'] - row['close_lag']) / row['close_lag']) * 100
        
    elif problem == 'VWAP':
        if row['VWAP_lag'] == 0:
            result = 0
        else:
            result = ((row['VWAP'] - row['VWAP_lag']) / row['VWAP_lag']) * 100

    elif problem == 'index':
        result = ((row['index_close'] - row['index_close_lag']) / row['index_close_lag']) * 100
    
    else:
        result = ((row['industry_close'] - row['industry_close_lag']) / row['industry_close_lag']) * 100

    result = round(result, 2)

    return result


def get_technical_indicators(data, columns_dict, look_back=4, forward=1):

    d = data.sort_values(by='ts').reset_index(drop=True)
    lag = columns_dict['lag']
    for col in lag:
        d[f'{col}_lag'] = d[col].shift(1)

    d['VWAP_after'] = d['VWAP'].shift(-forward)
    d['Y_weekly'] = d.apply(label, axis=1)

    return_col = columns_dict['return']
    for col in return_col:
        d[f'{col}_return'] = d.apply(log_return, problem=col, axis=1)

    
    d[f'sd_{look_back}'] = d['VWAP'].rolling(window=look_back).std()/d['VWAP'].rolling(window=look_back).mean()

    for problem in ['VWAP', 'close']:
        d[f'index_{problem}_slope'] = d.apply(index_slope, problem=problem, axis=1)
        d[f'industry_{problem}_slope'] = d.apply(industry_slope, problem=problem, axis=1)

    
    d['TR'] = d.apply(TR, axis=1)
    d['ATR_weekly'] = d['TR'].ewm(span=look_back, min_periods=look_back-1).mean()
    
           
    d = price_volume_trend(d)    
    d = RSI(d, n=look_back)
    d = STO(d, nk=4, nD=1)
    d = d.replace([np.inf, -np.inf], 0)
    d = d[d.ts.dt.date < date(2019,9,23)]

    d = d.drop(columns=['TR', 'VWAP_lag', 'VWAP_after', 'close_lag', 'index_close_lag', 'industry_close_lag'])

    return d


def merge_daily(daily_data, weekly_data, weekly_feature):

    d = weekly_data[weekly_feature]
    df = pd.merge(d, daily_data, on=['ts', 'StockNo'], how='inner')

    return df
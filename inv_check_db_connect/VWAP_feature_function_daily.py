import numpy as np 
import pandas as pd 
import datetime
from datetime import datetime, timedelta, date
from scipy.stats import skew, kurtosis
import math
# import pywt 
# import copy


def TR(row):
    TR = max([(row["high"] - row["low"]), abs(row["high"] - row["VWAP_lag"]), abs(row["VWAP_lag"] - row["low"])])
    
    return TR

#Compute RSI
def RSI(df, n):  
    df = df.sort_values(by='ts').reset_index(drop=True)
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

    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    RSI = RSI.replace([np.inf, -np.inf], np.nan)
    RSI = RSI.fillna(0) 
    df = df.join(RSI) 

    return df


#Compute Stochastic Index
def STO(df, nk, nD, forward):  

    if forward == 1:
        df = df.sort_values(by='ts').reset_index(drop=True)
        SOk = pd.Series((df['VWAP'] - df['low'].rolling(nk).min()) / (df['high'].rolling(nk).max() - df['low'].rolling(nk).min()), name = 'SOk'+str(nk)+f'_{forward}') 
        SOk = SOk.replace([np.inf, -np.inf], np.nan)
        SOk = SOk.fillna(0)
        SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SOd'+str(nD))
        SOd = SOd.fillna(0)
        df = df.join(SOk)
        df = df.join(SOd)

    else:
        df = df.sort_values(by='ts').reset_index(drop=True)
        SOk = pd.Series((df[f'VWAP_day{forward}'] - df['low'].rolling(nk).min()) / (df['high'].rolling(nk).max() - df['low'].rolling(nk).min()), name = 'SOk'+str(nk)+f'_{forward}') 
        SOk = SOk.replace([np.inf, -np.inf], np.nan)
        SOk = SOk.fillna(0)
        SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SOd'+str(nD))
        SOd = SOd.fillna(0)
        df = df.join(SOk)
        df = df.join(SOd)
    
    return df


def price_volume_trend(data):

    d = data.sort_values(by='ts').reset_index(drop=True)
    d['pvt_current'] = d['vol'] * (d['VWAP'] - d['VWAP_lag']) / d['VWAP_lag']

    for item in ['index_', 'industry_']:
        d[f'{item}pvt_current'] = d[f'{item}vol'] * (d[f'{item}close'] - d[f'{item}close'].shift(1)) / d[f'{item}close'].shift(1)

    for buyer in ['foreign_', 'investment_', 'dealer_']:
        d[f'{buyer}pvt_current'] = d[f'{buyer}buy'] * (d['VWAP'] - d['VWAP_lag']) / d['VWAP_lag']

    feature = [f'{item}pvt_current' for item in ['index_', 'industry_', 'foreign_', 'investment_', 'dealer_', '']]
    pvt_feature = [f'{item}pvt' for item in ['index_', 'industry_', 'foreign_', 'investment_', 'dealer_', '']]
    
    for f in pvt_feature:
        d[f] = 0

    d.loc[:, pvt_feature] = d.loc[:, feature].rolling(window=20, min_periods=1).sum().values

    d = d.drop(columns=feature)

    return d


def buy_ratio(row, buyer):

    if row['vol'] != 0:
        ratio = row[f'{buyer}'] / row['vol']
        return ratio
    else:
        return 0



def index_slope(row, problem='close'):

    slope = row[f'{problem}_return'] - row['index_return']

    return slope


def industry_slope(row, problem='close'):

    slope = row[f'{problem}_return'] - row['industry_return']

    return slope


def continuous_day(row, data, buyer):

    df = data[data.ts.dt.date <= row['ts'].date()]
    direction = row[f'{buyer}'] - 0
    

    continuous = 0

    if direction > 0:
        for i in range(1, 16):
            if i > len(df):
                break
            elif df[f'{buyer}'].iloc[-i] * direction > 0:
                continuous += 1
            else:
                break
    
    else:
        for i in range(1, 16):
            if i > len(df):
                break
            elif df[f'{buyer}'].iloc[-i] * direction > 0:
                continuous -= 1
            else:
                break

    return continuous


def buy_indicator(row, buyer):

    if row[buyer] > 0:
        return 1
    else:
        return 0


def sell_indicator(row, buyer):

    if row[buyer] < 0:
        return 1
    else:
        return 0


def label(row):

    if row['VWAP_after'] - row['VWAP_day5'] > 0:
        return 1
    elif row['VWAP_after'] - row['VWAP_day5'] <= 0:
        return 0


def log_return(row, problem='close'):

    assert problem in ['close', 'VWAP', 'index', 'industry', 'VWAP_day5']

    if problem == 'close':
        result = ((row['close'] - row['close_lag']) / row['close_lag'] if row['close_lag'] != 0 else 0) * 100
        
    elif (problem == 'VWAP') or (problem == 'VWAP_day5') :
        result = ((row[problem] - row[f'{problem}_lag']) / row[f'{problem}_lag'] if row[f'{problem}_lag'] != 0 else 0) * 100

    elif problem == 'index':
        result = ((row['index_close'] - row['index_close_lag']) / row['index_close_lag'] if row['index_close_lag'] != 0 else 0) * 100
    
    else:
        result = ((row['industry_close'] - row['industry_close_lag']) / row['industry_close_lag'] if row['industry_close_lag'] != 0 else 0) * 100

    result = round(result, 2)

    return result



def get_technical_indicators(data, columns_dict, look_back=15, forward=1):

    d = data.sort_values(by='ts').reset_index(drop=True)

#     d['VWAP_after'] = d['VWAP'].shift(-forward)
    # d['Y'] = d.apply(label, axis=1)

    lag = columns_dict['lag']
    for col in lag:
        d[f'{col}_lag'] = d[col].shift(1)

    return_col = columns_dict['return']
    for col in return_col:
        d[f'{col}_return'] = d.apply(log_return, problem=col, axis=1)


    # d['ema_20'] = d['VWAP'].ewm(span=20, min_periods=19).mean()
    # d['ema_5'] = d['VWAP'].ewm(span=5, min_periods=4).mean()
    # d['MACD'] = d['ema_5'] - d['ema_20']
    # d['signal'] = d['MACD'].ewm(span=15, min_periods=14).mean()
    # d['MACD_diff'] = d['MACD'] - d['signal']


    for problem in ['VWAP', 'close', f'VWAP_day{forward}']:
        d[f'index_{problem}_slope'] = d.apply(index_slope, problem=problem, axis=1)
        d[f'industry_{problem}_slope'] = d.apply(industry_slope, problem=problem, axis=1)


    ratio_col = columns_dict['ratio']
    for i in [5, 10, 15]:
        for col in ratio_col:
            d[f'{col}_ratio_day{i}'] = d[col]/ d[col].rolling(window=i, min_periods=1).mean()


    momentum_col = columns_dict['momentum']
    for i in [1, 2, 3, 4, 5, 10, 15]:
        for col in momentum_col:
            d[f'{col}_momentum_day{i}'] = d[col]/d[col].shift(i)


    d['vol_VWAP_corr'] =  d['VWAP'].rolling(window=look_back).corr(d['vol'])   
    d['vol_volatile_corr'] = d['vol'].rolling(window=look_back).corr(d['high']/d['low'])
    d['index_vol_close_corr'] =  d['index_close'].rolling(window=look_back).corr(d['index_vol'])    
    d['index_vol_volatile_corr'] = d['index_vol'].rolling(window=look_back).corr(d['index_high']/d['index_low'])
    d['industry_vol_close_corr'] =  d['industry_close'].rolling(window=look_back).corr(d['industry_vol'])    
    d['industry_vol_volatile_corr'] = d['industry_vol'].rolling(window=look_back).corr(d['industry_high']/d['industry_low'])
    
    #Compute skewness and kurtosis
    for col in columns_dict['moment']:
        d[f'{col}_skew'] = d[col].rolling(window=look_back).apply(lambda x: skew(x))
        d[f'{col}_kurtosis'] = d[col].rolling(window=look_back).apply(lambda x: kurtosis(x))

    
    buyer = columns_dict['buyer']
    for buy in buyer:
        d[f'{buy}_ratio'] = d.apply(buy_ratio, buyer=buy, axis=1)
        d[f'{buy}_day3_ratio'] = d[buy].rolling(window=3, min_periods=1).sum() / d['vol'].rolling(window=3, min_periods=1).sum() 
        d[f'{buy}_week_ratio'] = d[buy].rolling(window=5, min_periods=1).sum() / d['vol'].rolling(window=5, min_periods=1).sum()
        d[f'{buy}_2week_ratio'] = d[buy].rolling(window=10, min_periods=1).sum() / d['vol'].rolling(window=10, min_periods=1).sum()
        d[f'{buy}_3week_ratio'] = d[buy].rolling(window=15, min_periods=1).sum() / d['vol'].rolling(window=15, min_periods=1).sum() 
        d[f'{buy}_continuous'] = d.apply(continuous_day, data=d, buyer=buy, axis=1)
        d[f'{buy}_buy_indicator'] = d.apply(buy_indicator, buyer=buy, axis=1)
        d[f'{buy}_sell_indicator'] = d.apply(sell_indicator, buyer=buy, axis=1)
        d[f'{buy}_buy_day_ratio'] = d[f'{buy}_buy_indicator'].rolling(window=look_back).sum() / 15
        d[f'{buy}_sell_day_ratio'] = d[f'{buy}_sell_indicator'].rolling(window=look_back).sum() / 15

    
    d['TR'] = d.apply(TR, axis=1)
    d['ATR'] = d['TR'].ewm(span=look_back, min_periods=look_back-1).mean()
    
           
    d = price_volume_trend(d)    
    d = RSI(d, n=look_back)
    d = STO(d, nk=5, nD=3, forward=1)
    d = d.replace([np.inf, -np.inf], 0)
    d = d.fillna(0)

    d = d.drop(columns=['VWAP_lag', 'close_lag', 'index_close_lag', 'industry_close_lag'])

    return d


def separate_engineering(data, columns_dict, look_back=15, forward=5):

    d = data.sort_values(by='ts')
    if len(d[d['eliminate'] != 0]) == 0:
        return get_technical_indicators(d, columns_dict, look_back=look_back, forward=forward)

    else:
        start_date = d[d['eliminate'] == 2]['ts'].tolist()
        start_date.sort(reverse=True)
        df_list = []

        for start in start_date:
            d1 = d[d['ts'] < start]
            if len(d1) != 0:
                result = get_technical_indicators(d1, columns_dict, look_back=look_back, forward=forward)
                df_list.append(result)

                d = d[d['ts'] >= start]
            else:
                continue

        if len(d) > 0:
            result = get_technical_indicators(d, columns_dict, look_back=look_back, forward=forward)
            df_list.append(result)

        df = pd.concat(df_list, axis=0)

        return df


def get_label(data, forward=5):
    d = data.sort_values(by='ts').reset_index(drop=True)

    d[f'VWAP_day{forward}'] = d['total'].rolling(window=forward, min_periods=1).sum()/d['vol'].rolling(window=forward,min_periods=1).sum()
    d[f'VWAP_day{forward}'] = d[f'VWAP_day{forward}'].replace([np.inf, -np.inf], np.nan)
    d[f'VWAP_day{forward}'] = d[f'VWAP_day{forward}'].interpolate(method='pad')

    d['VWAP_after'] = d['VWAP_day5'].shift(-forward)
    d['Y'] = d.apply(label, axis=1)
    
    return d
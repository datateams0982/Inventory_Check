import numpy as np 
import pandas as pd 
import datetime
from datetime import datetime, timedelta, date
from scipy.stats import skew, kurtosis
import json   
from pathlib import Path
import os


global config
config_path = Path(__file__).parent.parent / "config/feature_config.json"
if not os.path.exists(config_path):
    raise Exception(f'Configs not in this Directory: {config_path}')

with open(config_path, 'r') as fp:
    config = json.load(fp)


def RSI(df, rolling_window): 

    '''
    Compute Relative Strength Index
    Input: {'df': The dataframe containing stock OHLCV, 'rolling_window': time period wanted  (by day)}
    Output: The Input dataframe with a new column 'RSI_{rolling_window}'
    ''' 

    df = df.sort_values(by='ts').reset_index(drop=True)
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df['high'].iat[i+1] - df['high'].iat[i]
        DoMove = df['low'].iat[i] - df['low'].iat[i+1]
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
    PosDI = pd.Series(UpI.ewm(span = rolling_window, min_periods = rolling_window - 1).mean())  
    NegDI = pd.Series(DoI.ewm(span = rolling_window, min_periods = rolling_window - 1).mean())  

    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(rolling_window))  
    RSI = RSI.replace([np.inf, -np.inf], np.nan)
    RSI = RSI.fillna(0) 
    df = df.join(RSI) 

    return df


def STO(df, nk, nD, forward):  

    '''
    Compute Stochastic Oscillator
    Input: {'df': The dataframe containing stock OHLCV, 'nk, nD': time period wanted  (by day), forward: The days looking forward in the main function}
    Output: The Input dataframe with two new column 'SOk_{nk}', 'SOd_{nD}'
    ''' 

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


def price_volume_trend(df, rolling_window):

    '''
    Compute price volume trend including stock price, index, industry index, and foreign/investment/dealer
    Input: {'df': The dataframe containing stock/index/industry OHLCV and buying volume}
    Output: The Input dataframe with 6 new columns 
    ''' 

    df_pvt = df.sort_values(by='ts').reset_index(drop=True)
    df_pvt['pvt_current'] = df_pvt['vol'] * (df_pvt['VWAP'] - df_pvt['VWAP_lag']) / df_pvt['VWAP_lag']

    for item in ['index_', 'industry_']:
        df_pvt[f'{item}pvt_current'] = df_pvt[f'{item}vol'] * (df_pvt[f'{item}close'] - df_pvt[f'{item}close'].shift(1)) / df_pvt[f'{item}close'].shift(1)

    for buyer in ['foreign_', 'investment_', 'dealer_']:
        df_pvt[f'{buyer}pvt_current'] = df_pvt[f'{buyer}buy'] * (df_pvt['VWAP'] - df_pvt['VWAP_lag']) / df_pvt['VWAP_lag']

    feature = [f'{item}pvt_current' for item in ['index_', 'industry_', 'foreign_', 'investment_', 'dealer_', '']]
    pvt_feature = [f'{item}pvt' for item in ['index_', 'industry_', 'foreign_', 'investment_', 'dealer_', '']]
    
    for f in pvt_feature:
        df_pvt[f] = 0

    df_pvt.loc[:, pvt_feature] = df_pvt.loc[:, feature].rolling(window=rolling_window, min_periods=1).sum().values

    df_pvt = df_pvt.drop(columns=feature)

    return df_pvt


def buy_ratio(row, buyer):

    '''
    Compute the volume ratio corporation bought
    Input: {'row': row of dataframe, 'buyer': foreign/investment/dealer}
    Output: Buy ratio 
    ''' 

    assert buyer in ['foreign_buy', 'investment_buy', 'dealer_buy']

    if row['vol'] != 0:
        ratio = row[f'{buyer}'] / row['vol']
        return ratio
    else:
        return 0



def index_slope(row, problem='close', forward=5):

    '''
    Compute the slope between stock return and index return
    Input: {'row': row of dataframe, containing stock and index return, 'problem': close or VWAP or VWAP_day{forward}}
    Output: slope  
    ''' 

    assert problem in ['close', 'VWAP', f'VWAP_day{forward}']

    slope = row[f'{problem}_return'] - row['index_return']

    return slope


def industry_slope(row, problem='close', forward=5):

    '''
    Compute the slope between stock return and industry return
    Input: {'row': row of dataframe, containing stock and industry return, 'problem': close or VWAP or VWAP_day{forward}}
    Output: slope  
    ''' 

    assert problem in ['close', 'VWAP', f'VWAP_day{forward}']

    slope = row[f'{problem}_return'] - row['industry_return']

    return slope


def continuous_day(row, data, buyer):

    '''
    Compute the length of day buyer continuing buying/selling
    Input: {'row': row of dataframe, containing buyer's buy, 'data': original data where row comes from, 'buyer': foreign_buy/investment_buy/dealer_buy}
    Output: length of continuous days  
    ''' 

    assert buyer in ['foreign_buy', 'investment_buy', 'dealer_buy']

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

    '''
    Define whether the buyer is buying this day
    Input: {'row': row of dataframe, containing buyer's buy, 'buyer': foreign_buy/investment_buy/dealer_buy}
    Output: if buy 1 else 0  
    ''' 

    assert buyer in ['foreign_buy', 'investment_buy', 'dealer_buy']

    if row[buyer] > 0:
        return 1
    else:
        return 0


def sell_indicator(row, buyer):

    '''
    Define whether the buyer is selling this day
    Input: {'row': row of dataframe, containing buyer's buy, 'buyer': foreign_buy/investment_buy/dealer_buy}
    Output: if sell 1 else 0  
    ''' 

    assert buyer in ['foreign_buy', 'investment_buy', 'dealer_buy']

    if row[buyer] < 0:
        return 1
    else:
        return 0


def log_return(row, problem='close'):

    '''
    Computing close/VWAP/VWAP_day5/index/industry return
    Input: {'row': row of dataframe, containing the two prices wanted to compare}
    Output: return ratio 
    ''' 

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


def label(row):

    '''
    labeling, if price return > 0: return 1, else 0
    Input: {'row': row of dataframe, containing the two prices wanted to compare}
    Output: if positive return 1 else 0  
    ''' 

    if row['VWAP_after'] - row['VWAP_day5'] > 0:
        return 1
    elif row['VWAP_after'] - row['VWAP_day5'] <= 0:
        return 0


def get_features(data, columns_dict, look_back=15, forward=5):

    '''
    The main function used to create feature
    Input: {'data': original data frame, containing basic information, 'columns_dict': Recording each feature's requiring column, 
            'look_back': the days looking back, 'forward': days looking forward for labeling}
    Output: a new dataframe containing original information and new features 
    ''' 

    d = data.sort_values(by='ts').reset_index(drop=True)
    d['total'] = d['total'] * 1000
    # Computing n days VWAP
    d[f'VWAP_day{forward}'] = d['total'].rolling(window=forward, min_periods=1).sum()/d['vol'].rolling(window=forward,min_periods=1).sum()
    d[f'VWAP_day{forward}'] = d[f'VWAP_day{forward}'].replace([np.inf, -np.inf], np.nan)
    d[f'VWAP_day{forward}'] = d[f'VWAP_day{forward}'].interpolate(method='pad')

    # d['VWAP_after'] = d['VWAP_day5'].shift(-forward)
    # d['Y'] = d.apply(label, axis=1)

    # Preparing for return calculation
    lag = columns_dict['lag']
    for col in lag:
        d[f'{col}_lag'] = d[col].shift(1)


    # Calculating return
    return_col = columns_dict['return']
    for col in return_col:
        d[f'{col}_return'] = d.apply(log_return, problem=col, axis=1)

    d[f'VWAP_day{forward}_lag'] = d[f'VWAP_day{forward}'].shift(forward)
    d[f'VWAP_day{forward}_return_{forward}'] = d.apply(log_return, problem=f'VWAP_day{forward}', axis=1)


    # Calculate slope
    for problem in ['VWAP', 'close', f'VWAP_day{forward}']:
        d[f'index_{problem}_slope'] = d.apply(index_slope, problem=problem, forward=forward, axis=1)
        d[f'industry_{problem}_slope'] = d.apply(industry_slope, problem=problem, forward=forward, axis=1)


    # Calculate price ratio comparing to previous price average
    ratio_col = columns_dict['ratio']
    ratio_days = config['ratio_days']
    for i in ratio_days:
        for col in ratio_col:
            d[f'{col}_ratio_day{i}'] = d[col]/ d[col].rolling(window=i, min_periods=1).mean()


    # Calculate price ratio comparing to previous price 
    momentum_col = columns_dict['momentum']
    momentum_days = config['momentum_days']
    for i in momentum_days:
        for col in momentum_col:
            d[f'{col}_momentum_day{i}'] = d[col]/d[col].shift(i)


    # Calculate price/volatile and volume correlation
    d['vol_VWAP_corr'] =  d['VWAP'].rolling(window=look_back).corr(d['vol'])   
    d['vol_volatile_corr'] = d['vol'].rolling(window=look_back).corr(d['high']/d['low'])
    d['index_vol_close_corr'] =  d['index_close'].rolling(window=look_back).corr(d['index_vol'])    
    d['index_vol_volatile_corr'] = d['index_vol'].rolling(window=look_back).corr(d['index_high']/d['index_low'])
    d['industry_vol_close_corr'] =  d['industry_close'].rolling(window=look_back).corr(d['industry_vol'])    
    d['industry_vol_volatile_corr'] = d['industry_vol'].rolling(window=look_back).corr(d['industry_high']/d['industry_low'])
    
    # Calculate skewness and kurtosis
    for col in columns_dict['moment']:
        d[f'{col}_skew'] = d[col].rolling(window=look_back).apply(lambda x: skew(x))
        d[f'{col}_kurtosis'] = d[col].rolling(window=look_back).apply(lambda x: kurtosis(x))

    
    # Calculate chip-related feature
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
        d[f'{buy}_buy_day_ratio'] = d[f'{buy}_buy_indicator'].rolling(window=look_back).sum() / look_back
        d[f'{buy}_sell_day_ratio'] = d[f'{buy}_sell_indicator'].rolling(window=look_back).sum() / look_back

    
    # Calculate other technical indicators       
    d = price_volume_trend(d, rolling_window=config['pvt_window'])    
    d = RSI(d, rolling_window=config['RSI_window'])
    d = STO(d, nk=config['STO_nk'][0], nD=config['STO_nD'][0], forward=1)
    d = STO(d, nk=config['STO_nk'][1], nD=config['STO_nD'][1], forward=forward)

    # Replacing infinity and nulls with 0
    d = d.replace([np.inf, -np.inf], 0)
    d = d.fillna(0)

    d = d.drop(columns=['VWAP_lag', 'close_lag', 'index_close_lag', 'industry_close_lag'])

    return d


def read_feature_list(file_path, requirement='whole'):

    '''
    Reading the list containing the features need for model
    Input: {'file_path': json path containing feature list, 'requirement': which list of feature}
    Output: list of feature
    ''' 

    assert requirement in ['whole', 'technical', 'buy', 'index']

    d = []
    with open(file_path, 'r') as fp:
        d = json.load(fp)
    
    feature_list = d[requirement]

    return feature_list


  
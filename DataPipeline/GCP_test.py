import numpy as np 
import pandas as pd 
import datetime
from datetime import datetime, timedelta, date
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm_notebook as tqdm
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
import calendar
import math
import pywt 
import copy
from multiprocessing import Pool
from functools import partial

#Compute True Range
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

    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    RSI = RSI.fillna(0) 

    return RSI


#Compute Stochastic Index
def STO(df, nk, nD):  
    df = df.reset_index(drop=True)
    SOk = pd.Series((df['close'] - df['low'].rolling(nk).min()) / (df['high'].rolling(nk).max() - df['low'].rolling(nk).min()), name = 'SO'+str(nk)) 
    SOk = SOk.fillna(0)
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO'+str(nD))
    SOd = SOd.fillna(0)
    
    return [SOk, SOd]

#Compute VWAP
def VWAP(row):  
    if row['vol'] == 0:
        return 0
    else:
        vwap = row['total']/row['vol']
        return vwap


def price_volume_trend(row):

    pvt = row['vol'] * (row['close'] - row['close_lag']) / row['close_lag']
        
    return pvt



#Denoise
def WT(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
    
    '''
    WT: Wavelet Transformation Function
    index_list: Input Sequence;
   
    lv: Decomposing Level；
 
    wavefunc: Function of Wavelet, 'db4' default；
    
    m, n: Level of Threshold Processing
   
    '''
   
    # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function 

    # Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold

    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
            
    denoised_index = np.sum(coeff, axis=0)   
        
    if plot:     
        data.plot(figsize=(10,5))
        plt.title(f'Level_{lv}')
   
    return denoised_index



def denoise_feature(data, columns_dict):

    d = data.sort_values(by='ts').reset_index(drop=True)
    d['origin_close'] = d['close']
    d = d.sort_values(by='ts').reset_index(drop=True)

    denoise_level = [[1200, 3], [4000, 4]]

    index_col = columns_dict['index']
    price_col = columns_dict['price']
    vol_col = columns_dict['vol']

    for item in denoise_level:
        if len(d) == 0:
            break
        if len(d) <= item[0]:
            level = item[1]
            for col in index_col:
                d[col] = WT(d[col], lv=level, n=level)
            for price in price_col:
                d[price] = WT(d[price], lv=level, n=level)
            for vol in vol_col:   
                if len(d[d[vol] != 0]) == 0:
                    continue
                d.loc[d[d[vol] != 0].index.tolist(), vol] = WT(d.loc[d[d[vol] != 0].index.tolist(), vol], lv=level, n=level)
            break
        else:
            continue

    return d

def denoise_technical_indicator(data, columns_dict):

    dataset = data.sort_values(by='ts').reset_index(drop=True)
    dataset['origin_close'] = dataset['close']
    dataset = dataset.sort_values(by='ts').reset_index(drop=True)

    denoise_level = [[500, 2], [1200, 3], [4000, 4]]

    index_col = columns_dict['index']
    price_col = columns_dict['price']
    vol_col = columns_dict['vol']

    for item in denoise_level:
        if len(dataset) == 0:
            break
        if len(dataset) <= item[0]:
            level = item[1]
            for col in index_col:
                dataset[col] = WT(dataset[col], lv=level, n=level)
            for price in price_col:
                dataset[price] = WT(dataset[price], lv=level, n=level)
            for vol in vol_col:    
                dataset.loc[dataset[dataset[vol] != 0].index.tolist(), vol] = WT(dataset.loc[dataset[dataset[vol] != 0].index.tolist(), vol], lv=level, n=level)
            break
        else:
            continue

    
    dataset['close_lag'] = dataset['close'].shift(1)
    d = dataset.iloc[-1]
    # Create 7 and 21 days Moving Average
    d['ma7'] = dataset['close'].iloc[-7:].mean()
    d['ma21'] = dataset['close'].iloc[-21:].mean()
    
    # Create MACD
    d['ema26'] = dataset['close'].iloc[-26:].ewm(span=26, min_periods=25).mean().iloc[-1]
    d['ema12'] = dataset['close'].iloc[-12:].ewm(span=12, min_periods=11).mean().iloc[-1]
    d['MACD'] = (d['ema12']-d['ema26'])
    # Create Bollinger Bands
    d['20sd'] = dataset['close'].iloc[-20:].std()
    d['upper_band'] = d['ma21'] + (d['20sd']*2)
    d['lower_band'] = d['ma21'] - (d['20sd']*2)

    corr_col = columns_dict['corr']
    for col in corr_col:
        d[f'{col}_corr'] = dataset['close'].iloc[-20:].corr(dataset[col].iloc[-20:])
    
    #Compute skewness and kurtosis
    skew_col = columns_dict['moment']
    for col in skew_col:
        d[f'{col}_skew'] = skew(dataset[col].iloc[-20:])
        d[f'{col}_kurtosis'] = kurtosis(dataset[col].iloc[-20:])

    #Compute PVT
    dataset['pvt_current'] = dataset.apply(price_volume_trend, axis=1)
    dataset['pvt'] = dataset['pvt_current'] + dataset['pvt_current'].shift(1)
    d['pvt'] = dataset['pvt'].iloc[-1]
    
    # Create True Range
    dataset['TR'] = dataset.apply(TR, axis=1)

    d['TR'] = dataset['TR'].iloc[-1]
    d['ATR'] = dataset['TR'].iloc[-15:].ewm(span=15).mean().iloc[-1]

    price_col = columns_dict['price'] + columns_dict['index']
    for col in price_col:
        if d[col] < 0:
            d[col] == 0
    
    # Create Reletive Strength Index
    d['RSI_15'] = RSI(dataset, n=15).iloc[-1]
    
    # Create Stochastic Oscillator
    STO_value = STO(dataset.iloc[-5:], nk=5, nD=3)
    d['SOk5'] = STO_value[0].iloc[-1]
    d['SOd3'] = STO_value[1].iloc[-1]

    return d


def get_technical_indicators_rolling_denoise_computation(data, columns_dict, lookback_days, SplitDate=date(2017,9,1)):
    data = data.sort_values(by='ts').reset_index(drop=True)
    if len(data) < lookback_days:
        return [False, data]

    data['VWAP'] = data.apply(VWAP, axis=1)
       
    d_list = []
    for i in range(lookback_days,len(data),1):
        date_end = data['ts'].iloc[i]
        df = data[data.ts.dt.date <= date_end][-lookback_days:]
        d = denoise_technical_indicator(df, columns_dict)
        d_list.append(pd.DataFrame(d.iloc[-1]).transpose())

    dataset = pd.concat(d_list, axis=0)
    dataset = dataset.drop(columns=['close_lag', '20sd'])

    return [True, dataset]


def get_technical_indicators_rolling_denoise_computation(data, columns_dict, lookback_days, SplitDate=date(2017,9,1)):
    data = data.sort_values(by='ts').reset_index(drop=True)
    if len(data) < lookback_days:
        return [False, data]

    data['VWAP'] = data.apply(VWAP, axis=1)
       
    d_list = []
    for i in range(lookback_days,len(data),1):
        date_end = data['ts'].iloc[i]
        df = data[data.ts.dt.date <= date_end][-lookback_days:]
        d = denoise_technical_indicator(df, columns_dict)
        d_list.append(pd.DataFrame(d.iloc[-1]).transpose())

    dataset = pd.concat(d_list, axis=0)
    dataset = dataset.drop(columns=['close_lag', '20sd'])

    return [True, dataset]


def get_technical_indicators_rolling_denoise_traintestsplit(data, columns_dict, lookback_days=250, SplitDate=date(2017,9,1)):
    data = data.sort_values(by='ts').reset_index(drop=True)
    if len(data) < lookback_days:
        return [False, data]

    data['VWAP'] = data.apply(VWAP, axis=1)

    d1 = data[data.ts.dt.date < SplitDate]
    d2 = data[data.ts.dt.date >= SplitDate]
    if len(d1) == 0:
        d_list = []
        for i in range(lookback_days,len(d2),1):
            date_end = d2['ts'].iloc[i]
            df = data[data.ts.dt.date <= date_end]
            d = denoise_technical_indicator(df, columns_dict)
            d_list.append(pd.DataFrame(d).transpose())

        dataset = pd.concat(d_list, axis=0)
        dataset = dataset.drop(columns=['close_lag', '20sd'])

        return[True, dataset]


    dataset = denoise_feature(d1, columns_dict)
    dataset['close_lag'] = dataset['close'].shift(1)
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['close'].rolling(window=21).mean()
    
    # Create MACD
    dataset['ema26'] = dataset['close'].ewm(span=26, min_periods=25).mean()
    dataset['ema12'] = dataset['close'].ewm(span=12, min_periods=11).mean()
    dataset['MACD'] = (dataset['ema12']-dataset['ema26'])
    # Create Bollinger Bands
    dataset['20sd'] = dataset['close'].rolling(window=20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

    corr_col = columns_dict['corr']
    for col in corr_col:
        dataset[f'{col}_corr'] = dataset['close'].rolling(20).corr(dataset[col])
    
    #Compute skewness and kurtosis
    skew_col = columns_dict['moment']
    for col in skew_col:
        dataset[f'{col}_skew'] = dataset[col].rolling(window=20).apply(lambda x: skew(x))
        dataset[f'{col}_kurtosis'] = dataset[col].rolling(window=20).apply(lambda x: kurtosis(x))

    #Compute PVT
    dataset['pvt_current'] = dataset.apply(price_volume_trend, axis=1)
    dataset['pvt'] = dataset['pvt_current'] + dataset['pvt_current'].shift(1)
    
    # Create True Range
    dataset['TR'] = dataset.apply(TR, axis=1)
    dataset['ATR'] = dataset['TR'].ewm(span=15).mean()

    price_col = columns_dict['price'] + columns_dict['index']
    for col in price_col:
        dataset.loc[dataset[dataset[col] < 0].index.tolist(), col] = 0
    
    # Create Reletive Strength Index
    dataset['RSI_15'] = RSI(dataset, n=15)
    
    # Create Stochastic Oscillator
    STO_value = STO(dataset, nk=5, nD=3)
    dataset['SOk5'] = STO_value[0]
    dataset['SOd3'] = STO_value[1]
    
    d_list = []
    lookback = max(0, lookback_days-len(d1))
    for i in range(lookback,len(d2),1):
        date_end = d2['ts'].iloc[i]
        df = data[data.ts.dt.date <= date_end]
        d = denoise_technical_indicator(df, columns_dict)
        d_list.append(pd.DataFrame(d).transpose())

    df_test = pd.concat(d_list, axis=0)
    df = pd.concat([dataset, df_test], axis=0)
    df = df.drop(columns=['close_lag', '20sd'])

    return [True, df]



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
    
    return df


def TrainTestSplit(data, TrainDate, valDate):

    train_df = data[data['ts'].dt.date < TrainDate][data['ema26'].notnull()]
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    val_df = data[data['ts'].dt.date < valDate][data['ts'].dt.date >= TrainDate][data['ema26'].notnull()]
    val_df = val_df.replace([np.inf, -np.inf], np.nan)
    test_df = data[data['ts'].dt.date >= valDate][data['ema26'].notnull()]
    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    train_df, val_df, test_df = train_df.fillna(0), val_df.fillna(0), test_df.fillna(0)

    return train_df, val_df, test_df

def GetScalerParam(train_df, feature_list, industry_index=['industry_open', 'industry_high', 'industry_low', 'industry_close', 'industry_vol']):
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



def Scaler(data, Param_df, industry_reference, feature_list, industry_index=['industry_open', 'industry_high', 'industry_low', 'industry_close', 'industry_vol'], problem='minmax0'):

    assert problem in ['standardize', 'minmax0']

    industry_list = data['industry'].unique().tolist()
    df_list = []

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


def transform_whole_csv(X, Y, feature_list, lookback=20):

    X = np.array(X)
    X_new = X.reshape(-1, X.shape[1] * X.shape[2])

    column_list = [f'{feature}_day{i+1}' for i in range(lookback) for feature in feature_list]

    df = pd.DataFrame(data=X_new[0:,0:], index=[i for i in range(X_new.shape[0])], columns=column_list)
    df['Y'] = [item[2] for item in Y]
    df['ts'] = [item[0] for item in Y]
    df['StockNo'] = [item[1] for item in Y]

    return df


def main():

    data = pd.read_csv('D:\\庫存健診開發\\data\\processed\\TWSE_stock_processed.csv', converters={'ts': str, 'StockName': str, 'StockNo': str})
    data['ts'] = pd.to_datetime(data['ts'])

    train_df = pd.read_csv('D:\\庫存健診開發\\data\\Clustering\\train_cluster.csv', index_col=0)
    test_df = pd.read_csv('D:\\庫存健診開發\\data\\Clustering\\test_cluster.csv', index_col=0)

    columns_dict = {'index': ['index_open', 'index_high', 'index_low', 'index_close', 'index_vol',
                           'industry_open', 'industry_high', 'industry_low', 'industry_close',
                           'industry_vol'],
               'price': ['open', 'high', 'low', 'close'],
               'vol': ['vol', 'total', 'VWAP'],
               'corr': ['index_close', 'industry_close', 'foreign_buy', 'investment_buy', 'dealer_buy'],
               'moment': ['close', 'index_close', 'industry_close', 'foreign_buy', 'investment_buy', 'dealer_buy']}

    df_list = [group[1] for group in data.groupby(data['StockNo'])]

    output_list = []

    with Pool(processes=12) as pool:
        for i, x in enumerate(tqdm(pool.imap_unordered(partial(get_technical_indicators_rolling_denoise_computation, columns_dict=columns_dict, lookback_days=250), df_list), total=len(df_list)), 1):
            if x[0]:
                output_list.append(x[1])
            else:
                continue
                    
    df = pd.concat(output_list, axis=0)

    d = total_split(train_df, test_df, df)

    train_df, val_df, test_df = TrainTestSplit(data, date(2016,1,1), date(2017,9,1))
    feature_list = ['open', 'high', 'low', 'close', 'vol',
       'VWAP', 'index_open', 'index_high', 'index_low', 'index_close', 'index_vol',
        'industry_open', 'industry_high', 'industry_low', 'industry_close', 'industry_vol',
        'foreign_buy', 'investment_buy', 'dealer_buy',
       'ma7', 'ma21', 'ema26', 'ema12', 'MACD', 'upper_band',
       'lower_band', 'index_close_corr', 'industry_close_corr',
       'foreign_buy_corr', 'investment_buy_corr', 'dealer_buy_corr',
       'close_skew', 'close_kurtosis', 'index_close_skew',
       'index_close_kurtosis', 'industry_close_skew',
       'industry_close_kurtosis', 'foreign_buy_skew', 'foreign_buy_kurtosis',
       'investment_buy_skew', 'investment_buy_kurtosis', 'dealer_buy_skew',
       'dealer_buy_kurtosis','pvt', 'ATR', 'RSI_15', 'SO5', 'SO3', 'total', 'capital']

    param, industry_param = GetScalerParam(train_df, feature_list)

    train_df, val_df, test_df = Scaler(train_df, param, industry_param, feature_list), Scaler(val_df, param, industry_param, feature_list), Scaler(test_df, param, industry_param, feature_list)

    feature_list = ['open', 'high', 'low', 'close', 'vol',
       'VWAP', 'index_open', 'index_high', 'index_low', 'index_close', 'index_vol',
        'industry_open', 'industry_high', 'industry_low', 'industry_close', 'industry_vol',
        'foreign_buy', 'investment_buy', 'dealer_buy',
       'ma7', 'ma21', 'ema26', 'ema12', 'MACD', 'upper_band',
       'lower_band', 'index_close_corr', 'industry_close_corr',
       'foreign_buy_corr', 'investment_buy_corr', 'dealer_buy_corr',
       'close_skew', 'close_kurtosis', 'index_close_skew',
       'index_close_kurtosis', 'industry_close_skew',
       'industry_close_kurtosis', 'foreign_buy_skew', 'foreign_buy_kurtosis',
       'investment_buy_skew', 'investment_buy_kurtosis', 'dealer_buy_skew',
       'dealer_buy_kurtosis','pvt', 'ATR', 'RSI_15', 'SO5', 'SO3', 'total', 'capital',
        'industry_open_scaled', 'industry_high_scaled', 'industry_low_scaled', 'industry_close_scaled', 'industry_vol_scaled']

    df_list = [group[1] for group in train_df.groupby(train_df['StockNo'])]

    X_train, Y_train = [], []

    with Pool(processes=12) as pool:
        for i, x in enumerate(tqdm(pool.imap_unordered(partial(func.separate_elimination, lookback=15, forward=5, feature_list=feature_list), df_list), total=len(df_list)), 1):
            for xx in x[0]:
                X_train.append(xx.tolist())
            Y_train.extend(x[1])
                

    assert len(X_train) == len(Y_train)
    train_data = transform_whole_csv(X_train, Y_train, feature_list, lookback=15)
    train_data = pd.merge(train_data, train_df[['ts', 'StockNo', 'cluster']], on=['ts', 'StockNo'], how='left')
    



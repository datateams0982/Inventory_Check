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

#Compute True Range
def TR(row):
    TR = max([(row["high"] - row["low"]), abs(row["high"] - row["close_lag"]), abs(row["close_lag"] - row["low"])])
    
    return TR


#Compute RSI
def RSI(df, n):  
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
    df = df.join(RSI)  

    return df


#Compute Stochastic Index
def STO(df, nk, nD):  
    SOk = pd.Series((df['close'] - df['low'].rolling(nk).min()) / (df['high'].rolling(nk).max() - df['low'].rolling(nk).min()), name = 'SO'+str(nk)) 
    SOk = SOk.fillna(0)
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO'+str(nD))
    SOd = SOd.fillna(0)
    df = df.join(SOk)
    df = df.join(SOd)
    
    return df

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

    denoise_level = [[1200, 3], [1800, 4], [4000, 5]]

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
                d.loc[d[d[vol] != 0].index.tolist(), vol] = WT(d.loc[d[d[vol] != 0].index.tolist(), vol], lv=level, n=level)
            break
        else:
            continue


    return d




##Compute Technical Indicators    
def get_technical_indicators(data, columns_dict, SplitDate=date(2017,9,1), denoise=True):
    data = data.reset_index(drop=True).sort_values(by='ts')

    data['VWAP'] = data.apply(VWAP, axis=1)
       

    if denoise:
        d1 = data[data['ts'] < SplitDate].reset_index(drop=True)
        d2 = data[data['ts'] >= SplitDate].reset_index(drop=True)
        if len(d1) > 0:
            d1 = denoise_feature(d1, columns_dict)
            d2 = denoise_feature(d2, columns_dict)
            dataset = pd.concat([d1, d2], axis=0).reset_index(drop=True).sort_values(by='ts')

        else:
            dataset = denoise_feature(d2, columns_dict)
       
    
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
    dataset = RSI(dataset, n=15)
    
    # Create Stochastic Oscillator
    dataset = STO(dataset, nk=5, nD=3)

    dataset = dataset.drop(columns=['20sd', 'close_lag'])

    return dataset



def validation_split(X_train, Y_train, step=20):
    length = len(X_train)//step
    val_index = [(step * i) - 1 for i in range(1, length + 1, 1)]
    train_index = [i for i in range(len(X_train)) if i%step != step-1]
    valX = [X_train[i] for i in val_index]
    trainX = [X_train[i] for i in train_index]
    valY = [Y_train[i] for i in val_index]
    trainY = [Y_train[i] for i in train_index]
    
    return trainX, trainY, valX, valY  
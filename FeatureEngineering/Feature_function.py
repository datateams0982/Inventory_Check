import numpy as np 
import pandas as pd 
import datetime
from datetime import datetime
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm_notebook as tqdm
import calendar
import math
import pywt 
import copy

#Compute True Range
def TR(row):
    TR = max([(row["high"] - row["low"]), (row["high"] - row["close_lag"]), (row["close_lag"] - row["low"])])
    
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
    SOk = pd.Series((df['close'] - df['low'].rolling(nk).min()) / (df['high'].rolling(nk).max() - df['low'].rolling(nk).min()), name = 'SO%k'+str(nk)) 
    SOk = SOk.fillna(0)
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))
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


##Compute Technical Indicators    
def get_technical_indicators(data):
    dataset = data.copy().reset_index(drop=True)
    
    dataset['close_lag'] = dataset['close'].shift(1)
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['close'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['close'].ewm(span=26, min_periods=25).mean()
    dataset['12ema'] = dataset['close'].ewm(span=12, min_periods=11).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
    # Create Bollinger Bands
    dataset['20sd'] = dataset['close'].rolling(window=20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create True Range
    dataset['TR'] = dataset.apply(TR, axis=1)
    dataset['ATR'] = dataset['TR'].ewm(span=15).mean()
    
    # Create Reletive Strength Index
    dataset = RSI(dataset, n=15)
    
    # Create Stochastic Oscillator
    dataset = STO(dataset, nk=5, nD=3)

    # Create VWAP
    dataset['VWAP'] = dataset.apply(VWAP, axis=1)
    
    dataset = dataset.drop(columns=['close_lag', '20sd'])

    return dataset


##Compute Time Feature
#Transform Intra-monthly Period by linear function
def IntraMonth(row, month_dict):
    y, m, date = row["ts"].year, row["ts"].month, row["ts"].date()
    d = month_dict[y][m]
    total = len(d)
    day = month_dict[y][m][date]
    time = day/(total - 1)
    
    return time

#Transform Intra-monthly feature to categorical (5 categories)
def MonthPeriod(row):
    num = row["Intramonth"]
    if num < 0.2:
        return 1
    elif num < 0.4:
        return 2
    elif num < 0.6:
        return 3
    elif num < 0.8:
        return 4
    else:
        return 5

#Transform week information by linear function
def WeekTime(row):
    y, m, d, date = row["ts"].year, row["ts"].month, row["ts"].day, row["ts"].date()
    week = date.isocalendar()[1] - date.replace(day=1).isocalendar()[1] + 1

    if week < 0:
        if m == 1:
            week = date.isocalendar()[1] + 1
        else:
            week = date.replace(day = d-7).isocalendar()[1] - date.replace(day=1).isocalendar()[1] + 2
        
    total = len(calendar.monthcalendar(y, m))
    week_time = (week - 1)/(total - 1)

    return week_time

#Transform Month to Categorical Feature
def OneHotMonth(data):
    onehot_encoder = OneHotEncoder(sparse=False)
    m = np.array(data['Month']).reshape(len(data['Month']), 1)
    month = onehot_encoder.fit_transform(m)

    data = pd.concat([data, pd.DataFrame(month, columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])], axis=1)

    return data

#Transform Weekday to Categorical Feature
def OneHotWeekday(data):
    onehot_encoder = OneHotEncoder(sparse=False)
    w = np.array(data['Weekday']).reshape(len(data['Weekday']), 1)
    week = onehot_encoder.fit_transform(w)
    if len(data['Weekday'].unique()) == 5:
        data = pd.concat([data, pd.DataFrame(week, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])], axis=1)

    else:
        data = pd.concat([data, pd.DataFrame(week, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])], axis=1)

    return data

#Transform Intra-month period to Categorical Feature
def OneHotPeriod(data):
    onehot_encoder = OneHotEncoder(sparse=False)
    w = np.array(data["IntramonthPeriod"]).reshape(len(data["IntramonthPeriod"]), 1)
    week = onehot_encoder.fit_transform(w)

    data = pd.concat([data, pd.DataFrame(week, columns=['First', 'Second', 'Third', 'Fourth', 'Fifth'])], axis=1)

    return data


def get_time_feature(df, month_dict):
    data = df.copy().reset_index(drop=True)

    data["Month"] = data["ts"].dt.month
    data["Weekday"] = data["ts"].dt.weekday + 1


    data["Intramonth"] = data.apply(IntraMonth, month_dict=month_dict, axis=1)
    data["IntramonthPeriod"] = data.apply(MonthPeriod, axis=1)
    data["WeekNum"] = data.apply(WeekTime, axis=1)

    return data


def categorical_transform(data):

    data = data.reset_index(drop=True)

    data = OneHotMonth(data)
    data = OneHotWeekday(data)
    data = OneHotPeriod(data)

    return data



#Weekly ATR
def WeeklylabelATR(row):
    if row['WeeklyReturn'] > row['ATR']:
        return 'up'
    elif row['WeeklyReturn'] < (-1) * row['ATR']:
        return 'down'
    else:
        return 'flat'

def Weeklylabel(row):
    if row['WeeklyReturn'] > 0:
        return 'up'
    elif row['WeeklyReturn'] < 0:
        return 'down'
    else:
        return 'flat'

def labelATR(row):
    if row['return'] > row['ATR']:
        return 'up'
    elif row['return'] < (-1) * row['ATR']:
        return 'down'
    else:
        return 'flat'

def label(row):
    if row['return'] > 0:
        return 'up'
    elif row['return'] < 0:
        return 'down'
    else:
        return 'flat'


def get_label(data):

    data['WeeklyReturn'] = data['close'] - data['close'].shift(5)
    df = data[data['26ema'].notnull()]

    if len(df) == 0:
        return df

    else:
        df['label_weekly_ATR'] = df.apply(WeeklylabelATR, axis=1)
        df['label_weekly'] = df.apply(Weeklylabel, axis=1)
        df['label_ATR'] = df.apply(labelATR, axis=1)
        df['label'] = df.apply(label, axis=1)
    
        return df

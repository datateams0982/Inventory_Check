import numpy as np 
import pandas as pd 
import os
import datetime
from datetime import datetime, timedelta, date
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm_notebook as tqdm
import calendar
import math
import pymssql as mssql
import time

#Send query to NsSQL
def send_query_to_MSSQL(query, db = 'ODS', timecost = True, showcol = True, showlen = True):
    """
    db: 選擇資料庫
    timecost = True: 統計此次查詢所花時間
    showcol = True: 顯示 table 欄位
    """

    tStart = time.time()
    if db == 'ODS':
        ods = mssql.connect(host = '128.110.13.89', user = '011553', password = 'Sino821031pac')
    if db == 'ODS_BK':
        ods = mssql.connect(host = '128.110.13.89', user = '011553', password = 'Sino821031pac')
    if db == 'CRM':
        ods = mssql.connect(host = '128.110.13.68', user = '011553', password = 'Sino821031pac')

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


def read_off_stock(filename, file_path, save_path):

    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{save_path}{filename[:4]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{save_path}{filename[:4]}.csv', converters={'股票代號': str})
    d = df[['股票代號', '終止日期']].rename(columns={'股票代號': 'StockNo', '終止日期': 'Off_Date'})
    d['Off_Date'] = d['Off_Date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def VWAP(row):  
    if (row['vol'] == 0) or (row['total'] == 0):
        return np.nan
    else:
        vwap = row['total']/row['vol']
        return vwap

def data_preprocess(filename, file_path, save_path):

    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{save_path}{filename[:8]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{save_path}{filename[:8]}.csv', converters={'日期': str, '股票代號': str})
    df = df[['日期', '股票代號', '股票名稱', '開盤價', '最高價', '最低價', '收盤價', '漲幅(%)', '振幅(%)', '成交量', '成交金額(千)', '總市值(億)']].rename(columns={'日期': 'ts', '股票代號': 'StockNo', '股票名稱': 'StockName', '開盤價': 'open', '最高價': 'high', '最低價': 'low', '收盤價': 'close', '漲幅(%)': 'return', '振幅(%)': 'volatility', '成交量': 'vol', '成交金額(千)': 'total', '總市值(億)': 'capital'})
    
    stock_list = df['StockNo'].unique().tolist()
    stock = [s for s in stock_list if (len(s) == 4) and (s[0] in [str(i) for i in range(1,10)])]
    
    d = df[df.StockNo.isin(stock)]
    
    d['VWAP'] = d.apply(VWAP, axis=1)
    d['ts'] = d['ts'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())

    index = df[df['StockName'] == '加權指數'][['ts', 'open', 'high', 'low', 'close', 'return', 'vol', 'total']].rename(columns={'open': 'index_open', 'high': 'index_high', 'low': 'index_low', 'close': 'index_close', 'return': 'index_return', 'vol': 'index_vol', 'total': 'index_total'})
    index = index.drop(columns=['index_total'])
    
    industry_list = [s for s in stock_list if s[:3] == 'TWB']
    industry_index = df[df.StockNo.isin(industry_list)][['ts', 'StockNo', 'open', 'high', 'low', 'close', 'return', 'vol', 'total']].rename(columns={'StockNo': 'reference', 'open': 'industry_open', 'high': 'industry_high', 'low': 'industry_low', 'close': 'industry_close', 'return': 'industry_return', 'vol': 'industry_vol', 'total': 'industry_total'})
    industry_index = industry_index.drop(columns=['industry_total'])

    return [d, index, industry_index]


def read_eliminate(filename, file_path, save_path):

    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{save_path}{filename[:4]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{save_path}{filename[:4]}.csv', converters={'股票代號': str})
    d = df[['年度', '股票代號', '股票名稱', '停止交易起始日', '開始買賣日期']].rename(columns={'年度': 'year', '股票代號': 'StockNo', '股票名稱': 'StockName', '停止交易起始日': 'Stop_date', '開始買賣日期': 'Restart_date'})
    d['Stop_date'] = d['Stop_date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    d['Restart_date'] = d['Restart_date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def read_inventory(filename, file_path, save_path):

    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{save_path}{filename[:8]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{save_path}{filename[:8]}.csv', converters={'股票代號': str})
    d = df[['日期', '股票代號', '外資買賣超', '投信買賣超', '自營商買賣超', '外資持股比率(%)', '投信持股比率(%)', '自營商持股比率(%)', '法人持股比率(%)']].rename(columns={'日期': 'ts', '股票代號': 'StockNo', '外資買賣超': 'foreign_buy', '投信買賣超': 'investment_buy', '自營商買賣超': 'dealer_buy', '外資持股比率(%)': 'foreign_ratio', '投信持股比率(%)': 'investment_ratio', '自營商持股比率(%)': 'dealer_ratio', '法人持股比率(%)': 'corporation_ratio'})
    d['ts'] = d['ts'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def read_fundamental(filename, file_path, save_path):

    directory = save_path
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{save_path}{filename[:4]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{save_path}{filename[:4]}.csv', converters={'股票代號': str, '上市日期': str})
    d = df[['年度', '股票代號', '上市日期', '產業名稱', '上市上櫃']].rename(columns={'年度': 'year', '股票代號': 'StockNo', '上市日期': 'On_Date', '產業名稱': 'industry', '上市上櫃': 'on'})
    d = d[d['on'] == 1][d['On_Date'] != '']
    d['On_Date'] = d['On_Date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def filter_stocks(data, fundamental, offstock):

    df = data[~data.StockNo.isin(offstock.StockNo.unique().tolist())]
    df = pd.merge(df, fundamental[['StockNo', 'On_Date']].drop_duplicates(), on='StockNo', how='inner')

    df = df[df.ts.dt.date > date(2007,7,1)][df.ts.dt.date < date(2019,9,24)]
    non_stock = fundamental[fundamental.industry == '存託憑證']['StockNo'].unique().tolist()
    df = df[~df.StockNo.isin(non_stock)]

    df.to_csv('D:\\庫存健診開發\\data\\processed\\TWSE_stock.csv', index=False)

    return df



def FillMissingTime(data, timedf, ondate, eliminate):

    data = data.sort_values(by='ts')
    Stock = data['StockNo'].unique()[0]
    timedf = timedf[timedf.ts >= data['On_Date'].iloc[0]]

    d = pd.merge(timedf, data, on="ts", how="left")

    if len(d) == 1:
        return d

    d = d.sort_values(by="ts")

    interpolate = ['open', 'high', 'low', 'close', 'VWAP', 'capital', 'StockNo', 'StockName']
    zero = ['total', 'vol']

    for col in interpolate:
        d[col] = d[col].interpolate(method="pad")
    
    for col in zero:
        d[col] = d[col].fillna(0)

    return d.sort_values(by='ts')


    # d['eliminate'] = 0
    # df = eliminate[eliminate.StockNo == Stock]
    # if len(df) == 0:
    #     d = d.drop(columns=['On_Date'])
    #     return d.sort_values(by='ts')

    # else:
    #     stop = df['Stop_date'].tolist()
    #     start = df['Restart_date'].tolist()
    #     d.loc[d[d.ts.isin(stop)].index.tolist(), 'eliminate'] = 1
    #     d.loc[d[d.ts.isin(start)].index.tolist(), 'eliminate'] = 2
    #     d = d.drop(columns=['On_Date'])

    #     return d.sort_values(by='ts')


def industry_reference(row):

    industry_dict = {'水泥工業': 'TWB11', '食品工業': 'TWB12', 
                    '塑膠工業': 'TWB13', '建材營建': 'TWB25', 
                    '化學工業': 'TWB30', '汽車工業': 'TWB22', 
                    '紡織纖維': 'TWB14', '貿易百貨': 'TWB29',
                    '其他': 'TWB99', '電子–電子通路': 'TWB38', 
                    '電子–半導體': 'TWB33', '電子–電子零組件': 'TWB37', 
                    '電機機械': 'TWB15', '鋼鐵工業': 'TWB20', 
                    '生技醫療': 'TWB31','電器電纜': 'TWB16', 
                    '電子–電腦及週邊設備': 'TWB34', '玻璃陶瓷': 'TWB18', 
                    '造紙工業': 'TWB19', '橡膠工業': 'TWB21', 
                    '航運業': 'TWB26', '電子–其他電子': 'TWB40',
                    '電子–通信網路': 'TWB36', '電子–光電': 'TWB35', 
                    '電子–資訊服務': 'TWB39', '油電燃氣': 'TWB32', 
                    '觀光事業': 'TWB27', '金融保險': 'TWB28'}

    if row['industry'] not in industry_dict:
        return np.nan

    reference = industry_dict[row['industry']]

    return reference

        

def merge_index(data, index, industry_index, fundamental):

    data['year'] = data.ts.dt.year
    d = pd.merge(data, fundamental[['year', 'StockNo', 'industry']], on=['year', 'StockNo'], how='left')
    if len(d.loc[d[d['year'] < 2009].index.tolist(), 'industry']) != 0:
        d.loc[d[d['year'] < 2009].index.tolist(), 'industry'] = d.loc[d[d['year'] == 2009].index.tolist()]['industry'].iloc[0]

    d['reference'] = d.apply(industry_reference, axis=1)
    df = pd.merge(d, index, on='ts', how='left')
    df = pd.merge(df, industry_index, on=['ts', 'reference'], how='left')
    df = df.drop(columns=['reference'])

    return df
    

def merge_inventory(data):

    d = data.sort_values(by='ts').reset_index(drop=True)
    for col in ['foreign_ratio', 'investment_ratio', 'dealer_ratio', 'corporation_ratio']:
        d[col] = d[col].interpolate(method="pad")

    d = d.fillna(0)

    return d

    
    






import numpy as np 
import pandas as pd 
import datetime
from datetime import datetime, date
from tqdm import tqdm_notebook as tqdm
import os
import codecs


def read_fundamental(filename, file_path):
    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{file_path}{filename[:4]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{file_path}{filename[:4]}.csv', converters={'股票代號': str})
    d = df[['年度', '股票代號', '股票名稱', '交易所普通股股本(千)', '上市日期']].rename(columns={'年度': 'year', '股票代號': 'StockNo', '股票名稱': 'StockName', '交易所普通股股本(千)': 'total_num', '上市日期': 'On_Date'})
    d['total_num'] = d['total_num'] * 100
    d['On_Date'] = d['On_Date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def read_eliminate(filename, file_path):
    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{file_path}{filename[:4]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{file_path}{filename[:4]}.csv', converters={'股票代號': str})
    df.loc[df[df['除權前股價'].isnull()].index.tolist(), '除權前股價'] = df.loc[df[df['除權前股價'].isnull()].index.tolist()]['除息前股價']
    df.loc[df[df['除權參考價'].isnull()].index.tolist(), '除權參考價'] = df.loc[df[df['除權參考價'].isnull()].index.tolist()]['除息參考價']
    d = df[['年度', '股票代號', '股票名稱', '停止交易起始日', '開始買賣日期', '除權前股價', '除權參考價']].rename(columns={'年度': 'year', '股票代號': 'StockNo', '股票名稱': 'StockName', '停止交易起始日': 'Stop_date', '開始買賣日期': 'Restart_date', '除權前股價': 'price_before', '除權參考價': 'price_after'})
    d['Stop_date'] = d['Stop_date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    d['Restart_date'] = d['Restart_date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d




def read_inventory(filename, file_path):
    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{file_path}{filename[:8]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{file_path}{filename[:8]}.csv', converters={'股票代號': str})
    d = df[['日期', '股票代號', '外資買賣超', '投信買賣超', '自營商買賣超']].rename(columns={'日期': 'ts', '股票代號': 'StockNo', '外資買賣超': 'foreign_buy', '投信買賣超': 'investment_buy', '自營商買賣超': 'dealer_buy'})
    d['ts'] = d['ts'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def read_fixed_price(filename, file_path):
    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{file_path}{filename[:8]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{file_path}{filename[:8]}.csv', converters={'股票代號': str})
    d = df[['日期', '股票代號', '還原收盤價']].rename(columns={'日期': 'ts', '股票代號': 'StockNo', '還原收盤價': 'fixed_close'})
    d['ts'] = d['ts'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d


def read_index(filename, file_path):

    df = pd.read_csv(f'{file_path}{filename}')
    date = datetime.strptime(filename[-14:-4], '%Y-%m-%d').date()
    reference = {}
    for col in df.columns.tolist():
        if col not in [ '日期', '時間', '未含金融保險股指數', '未含金融電子股指數', '未含電子股指數']:
            open_price = df.iloc[0][col]
            high = df[col].max()
            low = df[col].min()
            close = df.iloc[-1][col]
            reference[col] = [open_price, high, low, close]
        else:
            continue

    return [reference, date]
    

def read_off_stock(filename, file_path):
    
    with open(f'{file_path}{filename}', 'r', encoding="ansi") as fp, \
        open(f'{file_path}{filename[:8]}.csv', 'wb') as fw:
        for line in fp.readlines():
            fw.write(line.encode('utf-8'))

    df = pd.read_csv(f'{file_path}{filename[:8]}.csv', converters={'股票代號': str})
    d = df[['股票代號', '終止日期']].rename(columns={'股票代號': 'StockNo', '終止日期': 'Off_Date'})
    d['Off_Date'] = d['Off_Date'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    
    return d






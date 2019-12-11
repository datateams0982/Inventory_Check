import numpy as np 
import pandas as pd 
import os
import datetime
from datetime import datetime, timedelta, date
from cryptography.fernet import Fernet
import calendar
import pymssql as mssql
from pathlib import Path
from retry import retry
import json

global config
config_path = Path(__file__).parent.parent / "config/basic_config.json"
if not os.path.exists(config_path):
    raise Exception(f'Configs not in this Directory: {config_path}')

with open(config_path, 'r') as fp:
    config = json.load(fp)

#Send query to MsSQL
def send_query(query):
    
    '''
    Function sending query to ODS
    Input: Query(String)
    Output: Dataframe wanted 
    '''    

    encoding_path = Path(__file__).parent.parent / "config/mssqltip_bytes.bin"
    if not os.path.exists(encoding_path):
        raise Exception(f'Encoding Document not in this directory: {encoding_path}')
    
    key = config['db_pwd_key']
    cipher_suite = Fernet(key)
    with open(encoding_path, 'rb') as file_object:
        for line in file_object:
            encryptedpwd = line
    
    ods = mssql.connect(host = config['db_host'], 
                      user = config['db_user'], 
                      password = bytes(cipher_suite.decrypt(encryptedpwd)).decode("utf-8"), 
                      charset=config['db_charset'])

    odscur = ods.cursor(as_dict = True)
    odscur.execute(query)
    try:
        temp = odscur.fetchall()
    except:
        temp = []
        
    row_count = int(odscur.rowcount)
    df = pd.DataFrame(temp)
    ods.commit()
    ods.close()

    return df, row_count



def VWAP(row):  

    '''
    Function Computing Daily VWAP
    Input: row from dataframe containing volume and total
    Output: vwap
    '''

    if (row['vol'] == 0) or (row['total'] == 0):
        return np.nan

    else:
        vwap = row['total']/row['vol']
        return vwap


@retry(Exception, tries=config['query_retry']['tries'], delay=config['query_retry']['delay'])    
def stock_query(start_date, end_date):

    '''
    Function Querying data from db
    Input: The date wanted (date type)
    Output: A list of dataframe including stock, index, industry index
    '''

    # Precheck If tables in db have updated
    precheck_query = f'''SELECT 
                            MAX([DATE]) as max_date
                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                        '''

    max_date, _ = send_query(precheck_query)
    max_date = max_date['max_date'].iloc[0]
    max_date = datetime.strptime(str(max_date)[:4] + '-' + str(max_date)[4:6] + '-' + str(max_date)[6:], '%Y-%m-%d').date()

    if max_date < end_date:
        raise Exception('Data Not Updated')
    
    # Start query
    end_date = end_date.strftime('%Y%m%d')
    start_date = start_date.strftime('%Y%m%d')

    ETF_list = config['ETF_list']
    ETF_list = "('" + "','".join(map(str, ETF_list)) + "')"

    # Stock
    stock_subquery = f'''SELECT ts,
                        StockNo,
                        StockName, 
						[open],
                        [high],
                        [low],
                        [close], 
                        vol,
                        total,
                        capital,
                        VWAP, 
                        foreign_buy,
						investment_buy, 
                        dealer_buy,
                        foreign_ratio,
                        investment_ratio,
                        dealer_ratio,
                        corporation_ratio
                        FROM(
                            SELECT d.[DATE] AS ts,
                            d.[STOCK_ID] AS StockNo,
                            d.[STOCK_NAME] AS StockName,
                            TRY_CAST(d.[OPEN] AS FLOAT) AS [open], 
                            TRY_CAST(d.[HIGH] AS FLOAT) AS [high],
                            TRY_CAST(d.[LOW] AS FLOAT) AS [low],
                            TRY_CAST(d.[CLOSE] AS FLOAT) AS [close],
                            TRY_CAST(d.[VOLUME_SHARES] AS FLOAT) AS vol,
                            TRY_CAST(d.[AMOUNT] AS FLOAT) AS total,
                            TRY_CAST(d.[CAPITAL] AS FLOAT) AS capital,
                            TRY_CAST(d.[VWAP] AS FLOAT) AS VWAP,
                            TRY_CAST(e.[FOREIGN_VOL] AS FLOAT) AS foreign_buy,
                            TRY_CAST(e.[INVEST_VOL] AS FLOAT) AS investment_buy,
                            TRY_CAST(e.[DEALER_VOL] AS FLOAT) AS dealer_buy, 
                            TRY_CAST(e.[FOREIGN_INV_RATIO] AS FLOAT) AS foreign_ratio,
                            TRY_CAST(e.[INVEST_INV_RATIO] AS FLOAT) AS investment_ratio,
                            TRY_CAST(e.[DEALER_INV_RATIO] AS FLOAT) AS dealer_ratio,
                            TRY_CAST(e.[CORP_INV_RATIO] AS FLOAT) AS corporation_ratio
                    FROM 
                        (
                            SELECT *
                            FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                            WHERE 
                                DATE BETWEEN {start_date} AND {end_date}
                                AND STOCK_ID in {ETF_list}
                        ) d 
                        LEFT JOIN 
                        (    SELECT * 
                            FROM OpenData.dbo.CMONEY_DAILY_CORP_TXN_SUMMARY
                            WHERE 
                                DATE BETWEEN {start_date} AND {end_date}
                                AND STOCK_ID in {ETF_list}
                        ) e
                        ON d.STOCK_ID = e.STOCK_ID AND d.DATE = e.DATE
                        LEFT JOIN
                        (   SELECT STOCK_ID,
                                    MIN([DATE]) AS [START_DATE]
                            FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                            WHERE STOCK_ID in {ETF_list}
                            GROUP BY STOCK_ID)f
				        ON d.STOCK_ID = f.STOCK_ID
                        WHERE DATEADD(weekday , 60 , f.[START_DATE] ) <= CONVERT(varchar, '20190923', 23))p
                        '''
    
    stock_df, stock_row = send_query(stock_subquery)

    if len(stock_df) != stock_row:
        raise Exception("Stock data length doesn't match")

    stock_df['ts'] = pd.to_datetime(stock_df['ts'])
    
    # Index and Industry Index
    index_subquery = f'''SELECT [DATE] AS ts,
                                TRY_CAST([OPEN] AS FLOAT) AS [index_open], 
                                TRY_CAST([HIGH] AS FLOAT) AS [index_high],
                                TRY_CAST([LOW] AS FLOAT) AS [index_low],
                                TRY_CAST([CLOSE] AS FLOAT) AS [index_close],
                                TRY_CAST([VOLUME] AS FLOAT) AS index_vol
                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                        WHERE 
                            STOCK_ID = 'TWA00' 
                            AND DATE <= {end_date} and DATE >= {start_date}
                        '''
    
    
    index_df, index_row = send_query(index_subquery)
    if len(index_df) != index_row:
        raise Exception("Index data length doesn't match")

    index_df['ts'] = pd.to_datetime(index_df['ts'])
    
    return [stock_df, index_df]


def FillMissingTime(data, timedf, end_date):

    '''
    Filling Missing Time 
    Input: {'data': dataframe queried from db, 'timedf': Dataframe with all unique timestamps, 'end_date': The day wanted (same as query)}
    Output: Dataframe containing all timestamps and other information
    '''

    data = data.sort_values(by='ts')

    Stock = data['StockNo'].unique()[0]
    timedf = timedf[timedf.ts >= data['ts'].min()]

    # Check if the stock have exchanged enough days
    if len(timedf) <= config['min_exchange_days']:
        return [False]

    d = pd.merge(timedf, data, on="ts", how="left")
    d = d.sort_values(by="ts")

    interpolate = ['open', 'high', 'low', 'close', 'VWAP', 'capital', 'StockNo', 
                    'StockName', 'foreign_ratio', 'investment_ratio', 'dealer_ratio', 'corporation_ratio']
    zero = ['total', 'vol', 'foreign_buy', 'investment_buy', 'dealer_buy']

    for col in interpolate:
        d[col] = d[col].interpolate(method="pad")
        d[col] = d[col].fillna(0)
    
    for col in zero:
        d[col] = d[col].fillna(0)

    return [True, d.sort_values(by='ts')]


        

def merge_index(data, index):

    '''
    Merging stock data with index and industry index
    Input: {'data': Stock data after filling missing time, 'index': index data, 'industry_index': industry index data}
    Output: Merged dataframe
    '''
    
    df = pd.merge(data, index, on='ts', how='left')

    return df
    
    
    






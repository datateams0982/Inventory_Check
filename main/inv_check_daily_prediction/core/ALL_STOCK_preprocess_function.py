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
def stock_query(end_date):

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
    start_date = (end_date - timedelta(days=config['query_back'])).strftime('%Y%m%d')
    year = end_date.year
    month = end_date.month
    if month == 1:
        last_year = year - 1
        last_month = 12
    else:
        last_year = year
        last_month = month - 1

    last_month_start = date(last_year, last_month, 1).strftime('%Y%m%d')
    last_month_end = date(last_year, last_month, calendar.monthrange(last_year, last_month)[1]).strftime('%Y%m%d')
    end_date = end_date.strftime('%Y%m%d')

    # Stock
    stock_subquery = f'''SELECT price.ts,
                        price.StockNo,
                        price.StockName, 
                        price.[open],
                        price.[high],
                        price.[low],
                        price.[close], 
                        price.vol,
                        price.total,
                        price.capital,
                        price.VWAP, 
                        price.foreign_buy,
                        price.investment_buy, 
                        price.dealer_buy,
                        price.foreign_ratio,
                        price.investment_ratio,
                        price.dealer_ratio,
                        price.corporation_ratio,
                        price.industry,
                        price.On_Date,
                        reduction.START_TRADE_DATE as Restart_date,
						price.PctRank,
                        CASE WHEN START_TRADE_DATE IS NOT NULL THEN 1 ELSE 0 END AS eliminate
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
                            TRY_CAST(e.[CORP_INV_RATIO] AS FLOAT) AS corporation_ratio,
                            f.[INDUSTRY_ID] AS industry,
                            f.[EXCHANGE_DATE] AS [On_Date],
							TRY_CAST(p.PctRank AS FLOAT) AS PctRank
                    FROM 
                        (
                            SELECT *
                            FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                            WHERE 
                                DATE BETWEEN {start_date} AND {end_date}
                                AND LEN(STOCK_ID) = 4
                        ) d 
                        LEFT JOIN 
                        (    SELECT * 
                            FROM OpenData.dbo.CMONEY_DAILY_CORP_TXN_SUMMARY
                            WHERE 
                                DATE BETWEEN {start_date} AND {end_date}
                                and LEN(STOCK_ID) = 4
                        ) e
                        ON d.STOCK_ID = e.STOCK_ID AND d.DATE = e.DATE
                        LEFT JOIN 
                        (    SELECT
                                STOCK_ID,
                                LISTING_TYPE,
                                INDUSTRY_ID,
                                EXCHANGE_DATE
                            FROM OpenData.dbo.CMONEY_LISTED_COMPANY_INFO
                            WHERE 
                                [YEAR] = {str(year)}
                        ) f
                        ON d.STOCK_ID = f.STOCK_ID

						LEFT OUTER JOIN
						(	
                                  SELECT 
                                        STOCK_ID, 
                                        MONTHLY_AMT,
                                        PERCENT_RANK() OVER(ORDER BY MONTHLY_AMT) AS PctRank                    

                                    FROM(
                                        SELECT 
                                            a.STOCK_ID, 
                                            SUM(TRY_CAST(AMOUNT AS FLOAT)) as MONTHLY_AMT
                                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE  a
                                        LEFT JOIN 
                                        (    SELECT
                                                STOCK_ID,
                                                CONVERT(varchar, EXCHANGE_DATE, 23) as START_DATE,
                                                LISTING_TYPE,
                                                INDUSTRY_ID
                                            FROM OpenData.dbo.CMONEY_LISTED_COMPANY_INFO
                                            WHERE 
                                                [YEAR] = {str(year)}
                                        ) b
                                        ON a.STOCK_ID = b.STOCK_ID
                                        LEFT JOIN 
                                        (    SELECT
                                                STOCK_ID,
                                                TERMINATE_DATE
                                            FROM OpenData.dbo.CMONEY_DELISTED_COMPANY_INFO
                                            WHERE 
                                                [YEAR] = {str(year)}
                                        ) c
                                        ON a.STOCK_ID = c.STOCK_ID
                                        WHERE 
                                            a.DATE BETWEEN {last_month_start} AND {last_month_end}
                                            AND LEN(a.STOCK_ID) = 4
                                            AND LEFT(a.STOCK_ID, 1) between '1' and '9'
                                            AND LISTING_TYPE = 1
                                            AND TERMINATE_DATE IS NULL
                                            AND INDUSTRY_ID != '91'
                                            AND DATEADD(weekday , 60 , START_DATE ) <= CONVERT(varchar, {end_date}, 23)
                                        GROUP BY a.STOCK_ID
									) x
						) p
						ON d.STOCK_ID = p.STOCK_ID

                        WHERE 
                            EXISTS(
                                SELECT *
                                FROM(
                                    SELECT 
                                        STOCK_ID, 
                                        MONTHLY_AMT,
                                        PERCENT_RANK() OVER(ORDER BY MONTHLY_AMT) AS PctRank                    

                                    FROM(
                                        SELECT 
                                            a.STOCK_ID, 
                                            SUM(TRY_CAST(AMOUNT AS FLOAT)) as MONTHLY_AMT
                                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE  a
                                        LEFT JOIN 
                                        (    SELECT
                                                STOCK_ID,
                                                CONVERT(varchar, EXCHANGE_DATE, 23) as START_DATE,
                                                LISTING_TYPE,
                                                INDUSTRY_ID
                                            FROM OpenData.dbo.CMONEY_LISTED_COMPANY_INFO
                                            WHERE 
                                                [YEAR] = {str(year)}
                                        ) b
                                        ON a.STOCK_ID = b.STOCK_ID
                                        LEFT JOIN 
                                        (    SELECT
                                                STOCK_ID,
                                                TERMINATE_DATE
                                            FROM OpenData.dbo.CMONEY_DELISTED_COMPANY_INFO
                                            WHERE 
                                                [YEAR] = {str(year)}
                                        ) c
                                        ON a.STOCK_ID = c.STOCK_ID
                                        WHERE 
                                            a.DATE BETWEEN {last_month_start} AND {last_month_end}
                                            AND LEN(a.STOCK_ID) = 4
                                            AND LEFT(a.STOCK_ID, 1) between '1' and '9'
                                            AND LISTING_TYPE = 1
                                            AND TERMINATE_DATE IS NULL
                                            AND INDUSTRY_ID != '91'
                                            AND DATEADD(weekday , 60 , START_DATE ) <= CONVERT(varchar, {end_date}, 23)
                                        GROUP BY a.STOCK_ID
                                    ) p_rank
                                ) p_rank_filtered
                                WHERE PctRank >= 0
                                AND STOCK_ID = d.STOCK_ID
                            )) price
                            LEFT JOIN
                            (	SELECT STOCK_ID,
                                        START_TRADE_DATE
                                FROM OpenData.dbo.CMONEY_REDUCT_SUMMARY
                                WHERE 
                                    [YEAR] = {str(year)}
                            ) reduction
                            ON price.StockNo = reduction.STOCK_ID AND price.ts = reduction.START_TRADE_DATE
                        '''
    
    stock_df, stock_row = send_query(stock_subquery)

    if len(stock_df) != stock_row:
        raise Exception("Stock data length doesn't match")

    stock_df['ts'] = pd.to_datetime(stock_df['ts'])
    stock_df['On_Date'] = pd.to_datetime(stock_df['On_Date'])
    stock_df['Restart_date'] = pd.to_datetime(stock_df['Restart_date'])
    
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
    
    industry_subquery = f'''SELECT [DATE] AS ts,
                                [STOCK_ID] AS reference,
                                TRY_CAST([OPEN] AS FLOAT) AS [industry_open], 
                                TRY_CAST([HIGH] AS FLOAT) AS [industry_high],
                                TRY_CAST([LOW] AS FLOAT) AS [industry_low],
                                TRY_CAST([CLOSE] AS FLOAT) AS [industry_close],
                                TRY_CAST([VOLUME] AS FLOAT) AS industry_vol
                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                        WHERE 
                            SUBSTRING(STOCK_ID,1,3) = 'TWB' 
                            AND DATE <= {end_date} and DATE >= {start_date}
                        '''
    
    index_df, index_row = send_query(index_subquery)
    if len(index_df) != index_row:
        raise Exception("Index data length doesn't match")

    industry_df, industry_row= send_query(industry_subquery)
    if len(industry_df) != industry_row:
        raise Exception("Index data length doesn't match")

    index_df['ts'] = pd.to_datetime(index_df['ts'])
    industry_df['ts'] = pd.to_datetime(industry_df['ts'])
    
    return [stock_df, index_df, industry_df]


def FillMissingTime(data, timedf, end_date):

    '''
    Filling Missing Time 
    Input: {'data': dataframe queried from db, 'timedf': Dataframe with all unique timestamps, 'end_date': The day wanted (same as query)}
    Output: Dataframe containing all timestamps and other information
    '''

    data = data.sort_values(by='ts')

    # Check if the stock is trading today
    if data['ts'].iloc[-1].date() != end_date:
        return [False]

    Stock = data['StockNo'].unique()[0]
    timedf = timedf[timedf.ts >= data['On_Date'].iloc[0]]

    # Check if the stock have exchanged enough days
    if len(timedf) <= 1:
        return [False]

    # Check if today is 40 days after reduction
    if len(data[data.eliminate == 1]) != 0:
        restart_date = data[data.eliminate == 1]['ts'].iloc[-1]
        if len(data[data.ts.dt.date >= restart_date.date()]) <= 40:
            return [False]

    d = pd.merge(timedf, data, on="ts", how="left")
    d = d.sort_values(by="ts")

    interpolate = ['open', 'high', 'low', 'close', 'VWAP', 'capital', 'StockNo', 
                    'StockName', 'industry', 'foreign_ratio', 'investment_ratio', 'dealer_ratio', 'corporation_ratio']
    zero = ['total', 'vol', 'foreign_buy', 'investment_buy', 'dealer_buy']

    for col in interpolate:
        d[col] = d[col].interpolate(method="pad")
    
    for col in zero:
        d[col] = d[col].fillna(0)

    return [True, d.sort_values(by='ts')]


def industry_reference(row):

    '''
    Get Industry Index number inorder to merge with industry index
    Input: row in dataframe with industry
    Output: Industry index number
    '''

    industry_dict = {'1': 'TWB11', '2': 'TWB12', 
                    '3': 'TWB13', '14': 'TWB25', 
                    '21': 'TWB30', '12': 'TWB22', 
                    '4': 'TWB14', '18': 'TWB29',
                    '20': 'TWB99', '29': 'TWB38', 
                    '24': 'TWB33', '28': 'TWB37', 
                    '5': 'TWB15', '10': 'TWB20', 
                    '22': 'TWB31','6': 'TWB16', 
                    '25': 'TWB34', '8': 'TWB18', 
                    '9': 'TWB19', '11': 'TWB21', 
                    '15': 'TWB26', '31': 'TWB40',
                    '27': 'TWB36', '26': 'TWB35', 
                    '30': 'TWB39', '23': 'TWB32', 
                    '16': 'TWB27', '17': 'TWB28'}

    if row['industry'] not in industry_dict:
        return industry_dict['20']

    reference = industry_dict[row['industry']]

    return reference

        

def merge_index(data, index, industry_index):

    '''
    Merging stock data with index and industry index
    Input: {'data': Stock data after filling missing time, 'index': index data, 'industry_index': industry index data}
    Output: Merged dataframe
    '''
    
    data['reference'] = data.apply(industry_reference, axis=1)
    df = pd.merge(data, index, on='ts', how='left')
    df = pd.merge(df, industry_index, on=['ts', 'reference'], how='left')
    df = df.drop(columns=['reference'])

    return df
    
    
    






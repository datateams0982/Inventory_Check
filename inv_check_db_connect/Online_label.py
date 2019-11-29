import numpy as np 
import pandas as pd 
import os
import datetime
from datetime import datetime, timedelta, date
from tqdm import tqdm_notebook as tqdm
from cryptography.fernet import Fernet
import calendar
import math
import pymssql as mssql
import time
from retry import retry

def send_query(query):
    
    '''
    Function sending query to ODS
    '''
    encoding_path = os.path.join(os.sep, 'config', 'mssqltip_bytes.bin')
    if not os.path.exists(encoding_path):
        raise Exception(f'Encoding Document not in this directory: {encoding_path}')

    key = b'yFn37HvhJN2XPrV61AIk8eOG8MJw0lBXP2r32CJaPmk='
    cipher_suite = Fernet(key)
    with open(encoding_path, 'rb') as file_object:
        for line in file_object:
            encryptedpwd = line
    
    ods = mssql.connect(host = '128.110.13.89', 
                      user = 'OpenData', 
                      password = bytes(cipher_suite.decrypt(encryptedpwd)).decode("utf-8"), 
                      charset='utf8')

    odscur = ods.cursor(as_dict = True)
    odscur.execute(query)
    temp = odscur.fetchall()
    row_count = int(odscur.rowcount)
    df = pd.DataFrame(temp)
    odscur.close()

    return df, row_count

def stock_query(start_date, end_date):

    year = end_date.year
    start_date = start_date.strftime('%Y%m%d')
    end_date = end_date.strftime('%Y%m%d')

    subquery = f'''SELECT *, 
                    CASE WHEN FUT_VWAP_5D > PAST_VWAP_5D THEN 1 ELSE 0 END AS Y 
                FROM(
                    SELECT 
                    STOCK_ID AS StockNo,  
                    DATE AS ts, 
                    CASE WHEN PAST_VOLUME_5D = 0 THEN 0 ELSE PAST_AMT_5D/PAST_VOLUME_5D END as PAST_VWAP_5D, 
                    CASE WHEN FUT_VOLUME_5D = 0 THEN 0 ELSE FUT_AMT_5D/FUT_VOLUME_5D END as FUT_VWAP_5D
                    FROM
                    (
                        SELECT x.*,
                            SUM(VOLUME) over(partition by STOCK_ID order by STOCK_ID, DATE ROWS between 4 preceding and current row) as PAST_VOLUME_5D,
                            SUM(AMOUNT) over(partition by STOCK_ID order by STOCK_ID, DATE ROWS between 4 preceding and current row) as PAST_AMT_5D,
                            SUM(VOLUME) over(partition by STOCK_ID order by STOCK_ID, DATE ROWS between 1 following and 5 following) as FUT_VOLUME_5D,
                            SUM(AMOUNT) over(partition by STOCK_ID order by STOCK_ID, DATE ROWS between 1 following and 5 following) as FUT_AMT_5D
                        FROM (
                                    SELECT 
                                    i.DATE as [DATE],
                                    i.STOCK_ID as [STOCK_ID],
                                    case when j.VOLUME is null then 0.0 else [VOLUME] end as [VOLUME], 
                                    case when j.AMOUNT is null then 0.0 else [AMOUNT] end as [AMOUNT]
                                    FROM
                                    (SELECT * 
                                    FROM
                                    (SELECT distinct [DATE] 
                                    FROM
                                    OpenData.dbo.CMONEY_DAILY_CLOSE 
                                    where SUBSTRING([STOCK_ID], 1, 1) between '1' and '9'
                                    and [DATE] between {start_date} and {end_date}
                                    and len([STOCK_ID]) = 4
                                    ) a
                                    CROSS join
                                    (SELECT distinct b.STOCK_ID from(
                                                (SELECT
                                                        STOCK_ID,
                                                        LISTING_TYPE,
                                                        INDUSTRY_ID

                                                    FROM OpenData.dbo.CMONEY_LISTED_COMPANY_INFO
                                                    WHERE 
                                                        YEAR = {str(year)}
                                                ) b
                                                LEFT JOIN 
                                                (	SELECT
                                                        STOCK_ID,
                                                        TERMINATE_DATE
                                                    FROM OpenData.dbo.CMONEY_DELISTED_COMPANY_INFO
                                                    WHERE 
                                                        YEAR = {str(year)}
                                                ) c
                                                ON b.STOCK_ID = c.STOCK_ID
                                            )
                                            where LEN(b.STOCK_ID) = 4
                                                and LEFT(b.STOCK_ID, 1) between '1' and '9'
                                                and LISTING_TYPE = 1
                                                and TERMINATE_DATE IS NULL
                                                and INDUSTRY_ID != '91'
                                    ) b) i

                                    LEFT JOIN 
                                    (
                                        SELECT DATE, STOCK_ID, CAST(VOLUME AS FLOAT) AS VOLUME, CAST(AMOUNT AS FLOAT) AS AMOUNT
                                        FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                                    ) j

                                    on i.STOCK_ID = j.STOCK_ID and i.DATE = j.DATE) x) y) z
                                    '''

    stock_df, stock_row = send_query(subquery)
    stock_df['ts'] = stock_df['ts'].apply(lambda x: datetime.strptime(str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:], '%Y-%m-%d').date())
    stock_df['ts'] = pd.to_datetime(stock_df['ts'])

    return stock_df
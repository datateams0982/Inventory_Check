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

def send_query(query, db = 'OpenData', timecost = True, showcol = True, showlen = True):
    """
    db: 選擇資料庫
    timecost = True: 統計此次查詢所花時間
    showcol = True: 顯示 table 欄位
    """

    tStart = time.time()
    ods = mssql.connect(host = '128.110.13.89', user = '011553', password = 'Sino821031pac')

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


def stock_query(end_date):

    start_date = end_date - timedelta(days=100)

    subquery = f"""
                SELECT DATE AS ts, 
                    TRY_CAST(OPEN AS DECIMAL(7,2)) AS open,
                    TRY_CAST(HIGH AS DECIMAL(7,2)) AS high,
                    TRY_CAST(LOW AS DECIMAL(7,2)) AS low,
                    TRY_CAST(CLOSE AS DECIMAL(7,2)) AS close,
                    TRY_CAST(VOLUME AS DECIMAL(17,2)) AS vol,
                    TRY_CAST(AMOUNT AS DECIMAL(17,2)) AS total,
                    TRY_CAST(CAPITAL AS DECIMAL(17,2)) AS capital,
                    INDUSTRY_NAME AS industry
                    FROM 
                        (
                            SELECT * 
                            FROM OpenData.dbo.CMONEY_DAILY_CLOSE
                            WHERE 
                                DATE = '20191122'
                                AND LEN(STOCK_ID) = 4
                        ) d 
                        LEFT JOIN 
                        (    SELECT * 
                            FROM OpenData.dbo.CMONEY_DAILY_CORP_TXN_SUMMARY
                            WHERE 
                                DATE = '20191122'
                                and LEN(STOCK_ID) = 4
                        ) e
                        ON d.STOCK_ID = e.STOCK_ID
                        LEFT JOIN 
                        (    SELECT
                                STOCK_ID,
                                LISTING_TYPE,
                                INDUSTRY_ID 
                            FROM OpenData.dbo.CMONEY_LISTED_COMPANY_INFO
                            WHERE 
                                YEAR = '2019'
                        ) f
                        ON d.STOCK_ID = f.STOCK_ID
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
                                        SUM(TRY_CAST(AMOUNT AS DECIMAL(17,2))) as MONTHLY_AMT
                                    FROM OpenData.dbo.CMONEY_DAILY_CLOSE  a
                                    LEFT JOIN 
                                    (    SELECT
                                            STOCK_ID,
                                            LISTING_TYPE,
                                            INDUSTRY_ID
                                        FROM OpenData.dbo.CMONEY_LISTED_COMPANY_INFO
                                        WHERE 
                                            YEAR = '2019'
                                    ) b
                                    ON a.STOCK_ID = b.STOCK_ID
                                    LEFT JOIN 
                                    (    SELECT
                                            STOCK_ID,
                                            TERMINATE_DATE
                                        FROM OpenData.dbo.CMONEY_DELISTED_COMPANY_INFO
                                        WHERE 
                                            YEAR = '2019'
                                    ) c
                                    ON a.STOCK_ID = c.STOCK_ID
                                    WHERE 
                                        a.DATE between '20191001' and '20191031'
                                        and LEN(a.STOCK_ID) = 4
                                        and LEFT(a.STOCK_ID, 1) between '1' and '9'
                                        and LISTING_TYPE = 1
                                        and TERMINATE_DATE IS NULL
                                        and INDUSTRY_ID != '91'
                                    GROUP BY a.STOCK_ID
                                ) p_rank
                            ) p_rank_filtered
                            where PctRank >= 0.4
                            and STOCK_ID = d.STOCK_ID
                        )
                """



docker run -it --rm -p 8889:8888 -v D:\inv_check_db_connect\:/home/jovyan/workspace ods_jupyter

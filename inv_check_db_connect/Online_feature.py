import numpy as np 
import pandas as pd 
from cryptography.fernet import Fernet
from datetime import datetime, date, timedelta
from tqdm import tqdm_notebook as tqdm
import os
import pymssql as mssql
import calendar
import time
from multiprocessing import Pool    
from functools import partial
import requests
import os    
import json

import ALL_STOCK_preprocess_function as func
import VWAP_feature_function_rolling_week as func1


def main():

    #Read data from DB
    end_date = date.today()
    d = func.stock_query(end_date)
    stock, index, industry_index = d[0], d[1], d[2]


    #Fill Missing Time
    df_list = [group[1] for group in stock.groupby(stock['StockNo'])]
    timedf = pd.DataFrame(d[0]['ts'].unique(), columns=['ts'])
    timedf = timedf.sort_values(by='ts')
    output_list = []

    if __name__ == '__main__':
        with Pool(processes=5) as pool:
            for ___, x in enumerate(tqdm(pool.imap_unordered(partial(func.FillMissingTime, timedf=timedf, end_date=end_date), df_list), total=len(df_list)), 1):
                    if x[0]:
                        output_list.append(x[1])
                    else:
                        continue

    df = pd.concat(output_list)

    
    #Merge with index/industry data
    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    if __name__ == '__main__':
        with Pool(processes=5) as pool:
            for ___, x in enumerate(tqdm(pool.imap_unordered(partial(func.merge_index, index=index, industry_index=industry_index), df_list), total=len(df_list)), 1):
                    output_list.append(x)

    df = pd.concat(output_list)


    #Feature Engineering
    columns_dict = {'lag': ['index_close', 'industry_close', 'close', 'VWAP', 'VWAP_day5'],
            'return': ['index', 'industry', 'close', 'VWAP', 'VWAP_day5'],
            'ratio': ['index_close', 'index_vol', 'index_return', 'industry_close', 'industry_vol', 'industry_return', 'VWAP_return', 'close_return', 'vol', 'VWAP', 'VWAP_day5', 'VWAP_day5_return'],
            'momentum': ['index_close', 'index_vol', 'index_return', 'industry_close', 'industry_vol', 'industry_return', 'VWAP_return', 'close_return', 'vol', 'VWAP', 'VWAP_day5', 'VWAP_day5_return'],
            'moment': ['index_close', 'index_vol', 'industry_close', 'industry_vol', 'vol', 'VWAP', 'VWAP_day5', 'VWAP_day5_return'],
            'buyer': ['foreign_buy', 'investment_buy', 'dealer_buy']
            }

    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    if __name__ == '__main__':
        with Pool(processes=5) as pool:
            for ___, x in enumerate(tqdm(pool.imap_unordered(partial(func1.get_technical_indicators, columns_dict=columns_dict), df_list), total=len(df_list)), 1):
                output_list.append(x)

    df = pd.concat(output_list, axis=0)  

    #Get today's feature  
    df_last = df[df.ts.dt.date == end_date]
    feature = func1.read_feature_list('D:\\庫存健診開發\\feature_dict.json', requirement='whole')
    feature_df = df_last[feature]

    #prediction
    results=[]
    for i in tqdm(range(len(feature_df))):
        this_df = feature_df.iloc[[i]]
        result = func1.prediction(this_df)
        results.append(result)

    result_df = pd.DataFrame(results, columns=['StockNo','ts','Y_0_score','Y_1_score'])
    print(result_df)


if __name__ == '__main__':
    main()

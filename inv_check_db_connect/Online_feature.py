import numpy as np 
import pandas as pd 
from datetime import datetime, date, timedelta
from tqdm import tqdm_notebook as tqdm
import os
import pymssql as mssql
import calendar
import json
from multiprocessing import Pool    
from functools import partial 
import logging, traceback

import ALL_STOCK_preprocess_function as preprocess
import VWAP_feature_function_rolling_week as FeatureEngineering
import Prediction as Predict


# Logging Setting
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

log_path = os.path.join(os.sep, 'D:' + os.sep, '庫存健診開發', 'logging', f'log_{date.today()}.txt')
if not os.path.exists(log_path):
    with open(log_path, 'a'):
        os.utime(log_path, None)

fh = logging.FileHandler(log_path)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


def main():

    #Read data from DB
    end_date = date(2019,11,28)

    logging.info(f"Query Data From ODS.Opendata at {end_date}")
    try:
        d = preprocess.stock_query(end_date)
    except:
        logging.error(f'Query Failed \n {traceback.format_exc()}')
        raise Exception('Query Error')

    stock, index, industry_index = d[0], d[1], d[2]


    #Fill Missing Time
    df_list = [group[1] for group in stock.groupby(stock['StockNo'])]
    timedf = pd.DataFrame(d[0]['ts'].unique(), columns=['ts'])
    timedf = timedf.sort_values(by='ts')
    output_list = []


    logging.info(f"Filling Missing Time at {end_date}")
    try:
        if __name__ == '__main__':
            with Pool(processes=5) as pool:
                for x in pool.imap_unordered(partial(preprocess.FillMissingTime, timedf=timedf, end_date=end_date), df_list):
                        if x[0]:
                            output_list.append(x[1])
                        else:
                            continue

    except:
        logging.error(f'Failed when filling missing time \n {traceback.format_exc()}')
        raise Exception('Filling Time Error')

    df = pd.concat(output_list)

    
    #Merge with index/industry data
    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    logging.info(f'Merging Stock Data with Index Data at {end_date}')
    try:
        if __name__ == '__main__':
            with Pool(processes=5) as pool:
                for x in pool.imap_unordered(partial(preprocess.merge_index, index=index, industry_index=industry_index), df_list):
                        output_list.append(x)
    
        df = pd.concat(output_list)

    except:
        logging.error(f'Failed when merging index data \n {traceback.format_exc()}')
        raise Exception('Merging Index Error')



    #Reading column dict
    column_dict_path = 'columns_dict.json'

    if not os.path.exists(column_dict_path):
        logging.error(f'Failed when Reading Column Dict \n Column Dict not in this Directory: {column_dict_path}')
        raise Exception(f'Column Dict not in this Directory: {column_dict_path}')
        
    with open(column_dict_path, 'r') as fp:
        columns_dict = json.load(fp)

    #Feature Engineering
    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    logging.info(f'Feature Engineering at {end_date}')
    try:
        if __name__ == '__main__':
            with Pool(processes=5) as pool:
                for x in pool.imap_unordered(partial(FeatureEngineering.get_technical_indicators, columns_dict=columns_dict), df_list):
                    output_list.append(x)

    except:
        logging.error(f'Failed when Feature Engineering \n {traceback.format_exc()}')
        raise Exception('Feature Engineering Error')

    df = pd.concat(output_list, axis=0)  

    #Reading Feature List
    feature_dict_path = 'feature_dict.json'

    if not os.path.exists(feature_dict_path):
        logging.error(f'Failed when Reading Feature Dict \n Feature Dict not in this Directory: {feature_dict_path}')
        raise Exception(f'Feature Dict not in this Directory: {feature_dict_path}')
        
    feature = FeatureEngineering.read_feature_list(feature_dict_path, requirement='whole')
    df_last = df[df.ts.dt.date == end_date]
    feature_df = df_last[feature]


    #prediction
    results=[]

    logging.info(f'Predicting at {end_date}')
    try:
        feature_df['ts'] = feature_df['ts'].astype(str)
        for i in range(len(feature_df)):
            this_df = feature_df.iloc[[i]]
            result = Predict.prediction(this_df)
            results.append(result)
    except:
        logging.error(f'Failed when Predicting \n {traceback.format_exc()}')
        raise Exception('Predicting Error')

    result_df = pd.DataFrame(results, columns=['StockNo','ts','Y_0_score','Y_1_score'])
    

    #Reading Encoding File
    try:
        Predict.write_to_db(result_df, f'PREDICTION_{end_date}')    
    except:
        logging.error(f'Failed when Writing Prediction to Database \n {traceback.format_exc()}')
        raise Exception('Writing to Database Error')

    
    return

if __name__ == '__main__':
    main()

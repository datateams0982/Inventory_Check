import numpy as np 
import pandas as pd 
from datetime import datetime, date, timedelta
import os
import pymssql as mssql
import json
from multiprocessing import Pool    
from pathlib import Path
from functools import partial 
import logging, traceback
import sys

from core import ALL_STOCK_preprocess_function as preprocess
from core import VWAP_feature_function_rolling_week as FeatureEngineering
from core import Prediction as Predict
from core import exception_outbound 

## Documentation url: https://github.com/datateams0982/Inventory_Check/blob/Online/main/Documentation.md

## Load config
global config
config_path = Path(__file__).parent / "config/basic_config.json"
if not os.path.exists(config_path):
    exception_outbound.outbound(message=f'Exception while reading basic config: \nConfigs not in this Directory: {config_path}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
    raise Exception(f'Configs not in this Directory: {config_path}')

with open(config_path, 'r') as fp:
    config = json.load(fp)

# Logging Setting
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

global today_date
today_date = date.today().strftime('%Y%m%d')
global log_directory
log_directory = config['log_path']
log_path = Path(__file__).parent / f"{log_directory}log_{today_date}"
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

## Main Program
def main(start_date, end_date):

    if type(end_date) == str:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    feature_directory = config['feature_path']
    feature_path = Path(__file__).parent / f'{feature_directory}'

    # Remove previous log files and features
    logging.info(f"Removing previous log and results")
    try:
        remove_log_path = Path(__file__).parent / f'{log_directory}'
        remove_log_list = os.listdir(remove_log_path)
        remove_feature_list = os.listdir(feature_path)
        assert (len(remove_log_list) != 0) and (len(remove_feature_list) != 0)
        for log_file, feature_file in zip(remove_log_list, remove_feature_list):
            remove_log_file = remove_log_path + log_file
            remove_feature_file = feature_path + feature_file
            os.remove(remove_log_file)
            os.remove(remove_feature_file)

    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f"Can't Remove previous files")


    # Read data from DB
    logging.info(f"Query Data From ODS.Opendata at {today_date}")
    try:
        d = preprocess.stock_query(start_date, end_date)
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Query Failed \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while quering data from ODS.Opendata: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Query Error')

    stock, index, industry_index = d[0], d[1], d[2]


    # Fill Missing Time
    df_list = [group[1] for group in stock.groupby(stock['StockNo'])]
    timedf = pd.DataFrame(d[0]['ts'].unique(), columns=['ts'])
    timedf = timedf.sort_values(by='ts')
    output_list = []

    process_num = config['multiprocess']
    
    logging.info(f"Filling Missing Time at {today_date}")
    try:
        if __name__ == '__main__':
            with Pool(processes=process_num) as pool:
                for x in pool.imap_unordered(partial(preprocess.FillMissingTime, timedf=timedf, end_date=end_date), df_list):
                        if x[0]:
                            output_list.append(x[1])
                        else:
                            continue

    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when filling missing time \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while filling missing time: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Filling Time Error')

    df = pd.concat(output_list)

    
    # Merge with index/industry data
    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    logging.info(f'Merging Stock Data with Index Data at {today_date}')
    try:
        if __name__ == '__main__':
            with Pool(processes=process_num) as pool:
                for x in pool.imap_unordered(partial(preprocess.merge_index, index=index, industry_index=industry_index), df_list):
                        output_list.append(x)
    
        df = pd.concat(output_list)

    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when merging index data \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while merging index data: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Merging Index Error')


    # Reading column dict
    column_dict_path = Path(__file__).parent / "config/columns_dict.json"

    logging.info(f'Reading column dict at {today_date}')
    if not os.path.exists(column_dict_path):
        logging.error(f'Failed when Reading Column Dict \nColumn Dict not in this Directory: {column_dict_path}')
        exception_outbound.outbound(message=f'Exception while reading column dict: \nColumn Dict not in this Directory: {column_dict_path}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception(f'Column Dict not in this Directory: {column_dict_path}')
        
    with open(column_dict_path, 'r') as fp:
        columns_dict = json.load(fp)

    # Feature Engineering
    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    logging.info(f'Feature Engineering at {today_date}')
    try:
        if __name__ == '__main__':
            with Pool(processes=process_num) as pool:
                for x in pool.imap_unordered(partial(FeatureEngineering.separate_engineering, columns_dict=columns_dict, look_back=config['feature_lookback'], forward=config['feature_forward']), df_list):
                    output_list.append(x)

    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Feature Engineering \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while feature engineering: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Feature Engineering Error')

    df = pd.concat(output_list, axis=0)  
    df = df[df['SOk15_5'] != 0]
    df = df.fillna(0)

    # Split Training and Testing set
    logging.info(f'Split Training and Testing Data at {today_date}')
    try:
        train_df, test_df = FeatureEngineering.TrainTestSplit(df, SplitDate=config['split_date'])
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Splitting Training and Testing Data \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while Splitting Training and Testing Data: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Split Training and Testing Data Error')


    # Reading Feature List
    feature_dict_path = Path(__file__).parent / "config/feature_dict.json"

    logging.info(f'Reading feature list at {today_date}')
    if not os.path.exists(feature_dict_path):
        logging.error(f'Failed when Reading Feature Dict \nFeature Dict not in this Directory: {feature_dict_path}')
        exception_outbound.outbound(message=f'Exception while reading feature dict: \nFeature Dict not in this Directory: {feature_dict_path}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception(f'Feature Dict not in this Directory: {feature_dict_path}')
        
    feature = FeatureEngineering.read_feature_list(feature_dict_path, requirement=config['feature_requirement'])
    train_feature = feature + ['Y']
    test_feature = feature

    train_df = train_df[train_feature]
    test_df = test_df[test_feature]
    

    logging.info(f'Writing Feature to Local at {today_date}')
    try:
        train_df.to_csv(f'{feature_path}training_{today_date}.csv', index=False)
        test_df.to_csv(f'{feature_path}testing_{today_date}.csv', index=False)
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Writing Feature to Local \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while writing feature to local: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Write feature to local Error')
    
    logging.info(f'Data Preprocess Done at {today_date}')


    return

if __name__ == '__main__':
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    main(start_date, end_date)


    

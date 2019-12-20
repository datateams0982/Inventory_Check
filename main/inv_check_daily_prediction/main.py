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

global log_directory
log_directory = config['log_path']
log_path = Path(__file__).parent / f"{log_directory}{date.today().strftime('%Y%m%d')}"
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
def main(end_date=date.today(), force_engineering=False):

    if type(end_date) == str:
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    result_directory = config['result_path']
    feature_directory = config['feature_path']

    # Check if Feature Engineering is already done
    logging.info(f'Checking if Feature Engineering is already done at {end_date}')

    feature_path = Path(__file__).parent / f"{feature_directory}{end_date.strftime('%Y%m%d')}.csv"
    if os.path.exists(feature_path) and not force_engineering:

        feature_df = pd.read_csv(feature_path, converters={'ts': str, 'StockNo': str})
        if len(feature_df) >= config['check_length']:

            logging.info(f'Feature Engineering of {end_date} has done. Start to predict directly.')
            results=[]
            logging.info(f'Predicting at {end_date}')
            try:
                feature_df['ts'] = feature_df['ts'].astype(str)
                for i in range(len(feature_df)):
                    this_df = feature_df.iloc[[i]]
                    result = Predict.prediction(this_df)
                    results.append(result)

            except Exception as e:
                logging.error(f'Exception: {e}')
                logging.error(f'Failed when Predicting \n{traceback.format_exc()}')
                exception_outbound.outbound(message=f'Exception while predicting: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
                raise Exception('Predicting Error')

            result_df = pd.DataFrame(results, columns=['StockNo','ts','Y_0_score','Y_1_score'])
            
            # Write result to local
            logging.info(f'Writing to Local at {end_date}')
            result_path = Path(__file__).parent / f"{result_directory}{end_date.strftime('%Y%m%d')}.csv"
            try:
                result_df.to_csv(result_path, index=False)
            except Exception as e:
                logging.error(f'Exception: {e}')
                logging.error(f'Failed when Writing Prediction to Local \n{traceback.format_exc()}')
                exception_outbound.outbound(message=f'Exception while writing prediction to local: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
                raise Exception('Write to Local Error')


            # Writing result to DataBase
            logging.info(f'Writing to Database at {end_date}')
            try:
                Predict.write_to_db(result_df, config['writing_table'])    
            except Exception as e:
                logging.error(f'Exception: {e}')
                logging.error(f'Failed when Writing Prediction to Database \n{traceback.format_exc()}')
                exception_outbound.outbound(message=f'Exception while writing prediction to Database: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
                raise Exception('Write to Database Error')
            
            logging.info(f'Done at {end_date}')
            logging.info(f'Sending message and result to telegram at {end_date}')
            try:
                exception_outbound.outbound(message=f'Prediction at {end_date} Success; \nTime: {datetime.utcnow() + timedelta(hours=8)}', message_type='success', file_path=result_path)
            except Exception as e:
                logging.error(f'Exception: {e}')
                logging.error(f'Failed when sending message to telegram \n{traceback.format_exc()}')

            return

        else:
            logging.info(f'Feature Engineering at {end_date} has not done. Run the whole process.')
    else:
        logging.info(f'Feature Engineering at {end_date} has not done. Run the whole process.')


    # Remove log files and predictions 30 days ago
    logging.info(f"Removing previous log and results {config['preserve_days']} ago")
    try:
        remove_date = (end_date - timedelta(days=config['preserve_days'])).strftime('%Y%m%d')
        remove_log_path = Path(__file__).parent / f'{log_directory}{remove_date}'
        if not os.path.exists(remove_log_path):
            logging.warning(f'{remove_log_path} Not Exists or removed')
        else:
            os.remove(remove_log_path)

        remove_prediction_path = Path(__file__).parent / f'{result_directory}{remove_date}.csv'
        if not os.path.exists(remove_prediction_path):
            logging.warning(f'{remove_prediction_path} Not Exists or removed')
        else:
            os.remove(remove_prediction_path)

        remove_feature_path = Path(__file__).parent / f'{feature_directory}{remove_date}.csv'
        if not os.path.exists(remove_feature_path):
            logging.warning(f'{remove_feature_path} Not Exists or removed')
        else:
            os.remove(remove_feature_path)

    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f"Can't Remove previous files {config['preserve_days']} days ago")


    # Read data from DB
    logging.info(f"Query Data From ODS.Opendata at {end_date}")
    try:
        d = preprocess.stock_query(end_date)
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
    
    logging.info(f"Filling Missing Time at {end_date}")
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

    logging.info(f'Merging Stock Data with Index Data at {end_date}')
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

    logging.info(f'Reading column dict at {end_date}')
    if not os.path.exists(column_dict_path):
        logging.error(f'Failed when Reading Column Dict \nColumn Dict not in this Directory: {column_dict_path}')
        exception_outbound.outbound(message=f'Exception while reading column dict: \nColumn Dict not in this Directory: {column_dict_path}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception(f'Column Dict not in this Directory: {column_dict_path}')
        
    with open(column_dict_path, 'r') as fp:
        columns_dict = json.load(fp)

    # Feature Engineering
    df_list = [group[1] for group in df.groupby(df['StockNo'])]
    output_list = []

    logging.info(f'Feature Engineering at {end_date}')
    try:
        if __name__ == '__main__':
            with Pool(processes=process_num) as pool:
                for x in pool.imap_unordered(partial(FeatureEngineering.get_features, columns_dict=columns_dict, look_back=config['feature_lookback'], forward=config['feature_forward']), df_list):
                    output_list.append(x)

    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Feature Engineering \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while feature engineering: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Feature Engineering Error')

    df = pd.concat(output_list, axis=0)  


    # Reading Feature List
    feature_dict_path = Path(__file__).parent / "config/feature_dict.json"

    logging.info(f'Reading feature list at {end_date}')
    if not os.path.exists(feature_dict_path):
        logging.error(f'Failed when Reading Feature Dict \nFeature Dict not in this Directory: {feature_dict_path}')
        exception_outbound.outbound(message=f'Exception while reading feature dict: \nFeature Dict not in this Directory: {feature_dict_path}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception(f'Feature Dict not in this Directory: {feature_dict_path}')
        
    feature = FeatureEngineering.read_feature_list(feature_dict_path, requirement=config['feature_requirement'])
    df_last = df[df.ts.dt.date == end_date]
    feature_df = df_last[feature]

    logging.info(f'Writing Feature to Local at {end_date}')
    try:
        feature_df.to_csv(feature_path, index=False)
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Writing Feature to Local \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while writing feature to local: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Write feature to local Error')


    # prediction
    results=[]

    logging.info(f'Predicting at {end_date}')
    try:
        feature_df['ts'] = feature_df['ts'].astype(str)
        for i in range(len(feature_df)):
            this_df = feature_df.iloc[[i]]
            result = Predict.prediction(this_df)
            results.append(result)
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Predicting \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while predicting: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Predicting Error')

    result_df = pd.DataFrame(results, columns=['StockNo','ts','Y_0_score','Y_1_score'])
    
    # Write result to local
    logging.info(f'Writing to Local at {end_date}')
    result_path = Path(__file__).parent / f"{result_directory}{end_date.strftime('%Y%m%d')}.csv"
    try:
        result_df.to_csv(result_path, index=False)
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Writing Prediction to Local \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while writing prediction to local: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Write to Local Error')


    # Writing result to DataBase
    logging.info(f'Writing to Database at {end_date}')
    try:
        Predict.write_to_db(result_df, config['writing_table'])    
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when Writing Prediction to Database \n{traceback.format_exc()}')
        exception_outbound.outbound(message=f'Exception while writing prediction to Database: \n{e}; \nTime: {datetime.utcnow() + timedelta(hours=8)}')
        raise Exception('Write to Database Error')
    
    logging.info(f'Sending message and result to telegram at {end_date}')
    try:
        exception_outbound.outbound(message=f'Prediction at {end_date} Success; \nTime: {datetime.utcnow() + timedelta(hours=8)}', message_type='success', file_path=result_path)
    except Exception as e:
        logging.error(f'Exception: {e}')
        logging.error(f'Failed when sending message to telegram \n{traceback.format_exc()}')

    logging.info(f'Done at {end_date}')
    
    return

if __name__ == '__main__':
    if len(sys.argv) == 2:
        end_date = sys.argv[1]
        main(end_date)
    elif len(sys.argv) == 3:
        end_date = sys.argv[1]
        force_engineering = sys.argv[2]
        main(end_date, force_engineering)
    else:
        main()

    

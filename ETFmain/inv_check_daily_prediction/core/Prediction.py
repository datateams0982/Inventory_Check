import json
import requests
import os 
import pandas as pd 
from sqlalchemy import create_engine
from sqlalchemy.types import Date, String, Float
from cryptography.fernet import Fernet
from pathlib import Path
from retry import retry

from core import ALL_STOCK_preprocess_function as preprocess


global config
config_path = Path(__file__).parent.parent / "config/basic_config.json"
if not os.path.exists(config_path):
    raise Exception(f'Configs not in this Directory: {config_path}')

with open(config_path, 'r') as fp:
    config = json.load(fp)


@retry(tries=config['prediction_retry']['tries'], delay=config['prediction_retry']['delay'])
def prediction(df):

    df_json = df.to_dict(orient='records')
    predict_url = config['predict_url']
    x = requests.post(predict_url, data = json.dumps({'instances':df_json}))
    result = json.loads(x.text)
    prediction = result.get('predictions')
    scores = prediction[0].get('scores')
    this_result = list(df[['StockNo','ts']].values[0])+[scores[0],scores[1]]

    return this_result  


def write_to_db(df, table_name):

    encoding_path = Path(__file__).parent.parent / "config/mssqltip_bytes.bin"
    if not os.path.exists(encoding_path):
        raise Exception(f'Encoding Document not in this directory: {encoding_path}')
    
    key = config['db_pwd_key']
    cipher_suite = Fernet(key)
    with open(encoding_path, 'rb') as file_object:
        for line in file_object:
            encryptedpwd = line

    engine = create_engine("mssql+pymssql://{user}:{pw}@{host}/{db}"
                                .format(user=config['db_user'],
                                        pw=bytes(cipher_suite.decrypt(encryptedpwd)).decode("utf-8"),
                                        host = config['db_host'],
                                        db=config['db']))

    remove_query = f'''DELETE FROM OpenData.dbo.{config['writing_table']} WHERE ts = '{str(df.iloc[0]['ts'])}';'''
    _, _ = preprocess.send_query(remove_query)

    df[['ts', 'StockNo', 'Y_0_score', 'Y_1_score']].to_sql(config['writing_table'], 
                                                            con=engine, 
                                                            index=False, 
                                                            if_exists=config['writing_exists'], 
                                                            chunksize=config['writing_chunksize'],
                                                            dtype={"ts": Date(),"StockNo": String(),"Y_0_score": Float(), "Y_1_score": Float()})

    return
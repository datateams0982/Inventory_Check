import json
import requests
import os 
import pandas as pd 
from sqlalchemy import create_engine
from sqlalchemy.types import Date, String, Float
from cryptography.fernet import Fernet

def prediction(df):

    df_json = df.to_dict(orient='records')
    x = requests.post('http://128.110.238.61:8080/predict', data = json.dumps({'instances':df_json}))
    result = json.loads(x.text)
    prediction = result.get('predictions')
    scores = prediction[0].get('scores')
    this_result = list(df[['StockNo','ts']].values[0])+[scores[0],scores[1]]

    return this_result  


def write_to_db(df, table_name):

    encoding_path = os.path.join(os.sep, 'config', 'mssqltip_bytes.bin')
    if not os.path.exists(encoding_path):
        raise Exception('Encoding Document not in this directory')

    key = b'yFn37HvhJN2XPrV61AIk8eOG8MJw0lBXP2r32CJaPmk='
    cipher_suite = Fernet(key)
    with open(encoding_path, 'rb') as file_object:
        for line in file_object:
            encryptedpwd = line

    engine = create_engine("mssql+pymssql://{user}:{pw}@128.110.13.89/{db}"
                                .format(user="OpenData",
                                        pw=bytes(cipher_suite.decrypt(encryptedpwd)).decode("utf-8"),
                                        db="OpenData"))


    df[['ts', 'StockNo', 'Y_0_score', 'Y_1_score', 'Y']].to_sql(table_name, con= engine, index=False, if_exists='append', chunksize=1000, dtype={"ts": Date(),"StockNo": String(),"Y_0_score": Float(), "Y_1_score": Float(), 'Y': Float()})

    return
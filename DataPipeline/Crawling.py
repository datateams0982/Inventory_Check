import requests
import os
import json
import codecs
import pandas as pd
from datetime import date, timedelta
import time
from retrying import retry
import random


class TWSE_Crawler:

    def __init__(f=1):
        f 



@retry(stop_max_attempt_number=5, wait_fixed=5000)
def download_twse_csv(download_path,data_date):
	data_date_parsed = data_date.replace('-','')
	json_URL = f'https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={data_date_parsed}&type=ALLBUT0999&_=1569302830946'

	with requests.get(json_URL, stream=True) as r:
		if not r:
			print("Requests status is bad: {}. Raising exception.".format(r.status_code))
			r.raise_for_status()
			time.sleep(10)
		
		d = json.loads(r.content.decode('utf-8'))
		status = True

	if 'data9' in d:
		df = pd.DataFrame(d['data9']).iloc[:, [0,1,2,4,5,6,7,8]].rename(columns={0: 'StockNo', 1:'StockName', 2:'vol', 4:'total', 5:'open', 6:'high', 7:'low', 8:'close'})	
		df.to_csv(download_path, index=False)	
	elif 'data8' in d:
		df = pd.DataFrame(d['data8']).iloc[:, [0,1,2,4,5,6,7,8]].rename(columns={0: 'StockNo', 1:'StockName', 2:'vol', 4:'total', 5:'open', 6:'high', 7:'low', 8:'close'})
		df.to_csv(download_path, index=False)
	else:
		status = False

	return status
from tqdm import tqdm_notebook as tqdm
import numpy as np  
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import math
import os
from datetime import datetime, timedelta, date


class TrainProcess:

    def __init__(self, df_path, cluster_num, train_date, val_date, 
                    feature_list=['close', 'high', 'low', 'open',
                    'vol', 'VWAP', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band',
                    'lower_band', 'skew', 'kurtosis', 'pvt', 'ATR',
                    'RSI_15', 'SO%k5', 'SO%d3']):

        self._df = pd.read_csv(f'{df_path}Cluster_{cluster_num}.csv', converters={'ts': str, 'StockNo': str, 'StockName': str})
        self._df['ts'] = pd.to_datetime(self._df['ts'])
        self._traindate = train_date
        self._valdate = val_date
        self._train_df = self._df[self._df['ts'].dt.date < self._traindate][self._df['26ema'].notnull()]
        self._val_df = self._df[self._df['ts'].dt.date < self._valdate = val_date][self._df['ts'].dt.date >= self._traindate][self._df['26ema'].notnull()]
        self._test_df = self._df[self._df['ts'].dt.date >= self._valdate = val_date][self._df['26ema'].notnull()]
        self._train_scaled = []
        self._val_scaled = []
        self._test_scaled = []
        self._feature = feature_list

        if (not self._train_df[self._feature].isna().values.any()) or (not self._val_df[self._feature].isna().values.any()) or (not self._test_df[self._feature].isna().values.any()):
            raise ('Exist Missing Value in data')

    
    @property
    def train_df(self):
        return self._train_df

    @property
    def val_df(self):
        return self._val_df

    @property
    def test_df(self):
        return self._test_df

    @property
    def train_scaled(self):
        return self._train_scaled

    @property
    def val_scaled(self):
        return self._val_scaled

    @property
    def test_scaled(self):
        return self._test_scaled

    @property
    def feature(self):
        return self._feature


    def _GetScalerParam(self):

        self._paramdf = pd.DataFrame(columns=['max', 'min', 'std', 'mean'])
        self._paramdf['max'] = self._train_df[self._feature].max()
        self._paramdf['min'] = self._train_df[self._feature].min()


    def _Scaler(self):

        self._GetScalerParam()

        Max = self._paramdf['max']
        Min = self._paramdf['min']

        for f in self._feature:
            self._train_scaled[f] = (self._train_df[f] - Min.loc[f])/(Max.loc[f] - Min.loc[f])
            self._val_scaled[f] = (self._val_df[f] - Min.loc[f])/(Max.loc[f] - Min.loc[f])
            self._test_scaled[f] = (self._test_df[f] - Min.loc[f])/(Max.loc[f] - Min.loc[f])

    





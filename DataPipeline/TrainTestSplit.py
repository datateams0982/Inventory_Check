import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta, date
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
from functools import partial
import pickle


def create_dataset(dataset, lookback, forward, feature_list, problem='classification'):

    if problem.lower() not in ['regression', 'classification']:
        return 'Problem not exit'

    dataX = []
    dataY = []

    dataset = dataset.sort_values(by='ts').reset_index(drop=True)
    for i in range(0,dataset.shape[0]-lookback-forward,1):
        this_ds = dataset.iloc[i:i+lookback].copy()
        
        my_ds =  this_ds[feature_list]
                       
        dataX.append(np.array(my_ds))

        if problem == 'regression':
            y = dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo', 'origin_close']]
            dataY.append(np.array(y))

        else:
            price_return =  dataset.iloc[i + lookback + forward - 1]['origin_close'] - dataset.iloc[i + lookback - 1]['origin_close']
            if price_return > 0:
                y = 1
            else:
                y = 0

            Y =  dataset.iloc[i + lookback + forward - 1][['ts', 'StockNo']].tolist()
            Y.append(y)
            dataY.append(Y)

    return [dataX, dataY]


def separate_elimination(data, lookback, forward, feature_list, problem='classification'):
    d = data.copy()

    if len(d[d['eliminate'] != 0]) == 0:
        return create_dataset(d, lookback, forward, feature_list, problem='classification')

    start_date = d[d['eliminate'] == 2]['ts'].tolist()
    start_date.sort(reverse=True)
    dataX = []
    dataY = []
    for start in start_date:
        d1 = d[d['ts'] <= start]
        result = create_dataset(d1, lookback, forward, feature_list, problem='classification')
        dataX.extend(result[0])
        dataY.extend(result[1])

        d = d[d['ts'] > start]

    result = create_dataset(d, lookback, forward, feature_list, problem='classification')
    dataX.extend(result[0])
    dataY.extend(result[1])

    return [dataX, dataY]

class TrainProcess:

    def __init__(self, df_path, cluster_num, train_date, val_date, save_path, lookback=20, forward=5,
                    feature_list=['close', 'high', 'low', 'open',
                    'vol', 'VWAP', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band',
                    'lower_band', 'skew', 'kurtosis', 'pvt', 'ATR',
                    'RSI_15', 'SO%k5', 'SO%d3'], problem='classification'):

        self._df = pd.read_csv(f'{df_path}Cluster_{cluster_num}.csv', converters={'ts': str, 'StockNo': str, 'StockName': str})
        self._df['ts'] = pd.to_datetime(self._df['ts'])
        self._cluster = cluster_num
        self._traindate = train_date
        self._valdate = val_date
        self._train_df = self._df[self._df['ts'].dt.date < self._traindate][self._df['26ema'].notnull()]
        self._val_df = self._df[self._df['ts'].dt.date < self._valdate = val_date][self._df['ts'].dt.date >= self._traindate][self._df['26ema'].notnull()]
        self._test_df = self._df[self._df['ts'].dt.date >= self._valdate = val_date][self._df['26ema'].notnull()]
        self._train_scaled = []
        self._val_scaled = []
        self._test_scaled = []
        self._feature = feature_list
        self._lookback = lookback
        self._forward = forward
        self._problem = problem
        self._path = save_path


        if (self._train_df[self._feature].isna().values.any()) or (self._val_df[self._feature].isna().values.any()) or (self._test_df[self._feature].isna().values.any()):
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

    
    def data_transformation(self):

        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_test, self._Y_test = [], [], [], [], [], []

        self._GetScalerParam()
        self._Scaler()

        self._train_list = [group[1] for group in self._train_df.groupby(self._train_df['StockNo'])]
        self._val_list = [group[1] for group in self._val_df.groupby(self._val_df['StockNo'])]
        self._test_list = [group[1] for group in self._test_df.groupby(self._test_df['StockNo'])]

        with Pool(processes=12) as pool:
            for i, x in enumerate(tqdm(pool.imap_unordered(partial(separate_elimination, lookback=self._lookback, forward=self._forward, feature_list=self._feature, problem=self._problem), self._train_list), total=len(self._train_list)), 1):
                for xx in x[0]:
                    self.X_train.append(xx.tolist())
                self.Y_train.extend(x[1])

        with Pool(processes=12) as pool:
            for i, x in enumerate(tqdm(pool.imap_unordered(partial(separate_elimination, lookback=self._lookback, forward=self._forward, feature_list=self._feature, problem=self._problem), self._val_list), total=len(self._val_list)), 1):
                for xx in x[0]:
                    self.X_val.append(xx.tolist())
                self.Y_val.extend(x[1])

        with Pool(processes=12) as pool:
            for i, x in enumerate(tqdm(pool.imap_unordered(partial(separate_elimination, lookback=self._lookback, forward=self._forward, feature_list=self._feature, problem=self._problem), self._test_list), total=len(self._test_list)), 1):
                for xx in x[0]:
                    self.X_test.append(xx.tolist())
                self.Y_test.extend(x[1])

        
        if (not (len(self.X_train) == len(self.Y_train))) or (not(len(self.X_val) == len(self.Y_val))) or (not(len(self.X_test) == len(self.Y_test)):

            raise TypeError('Invalid Data Shape')


    def save_data(self):

        save_list = [self.X_trian, self.Y_train, self.X_val, self.Y_val, self.X_test, self.Y_test]

        with open(f'{self._path}Cluster_{self._cluster}_{self._problem}_{self._forward}day', 'wb') as fp:
            pickle.dump(save_list, fp)




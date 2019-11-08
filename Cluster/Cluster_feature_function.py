import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta, date
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import pywt
import copy
import matplotlib.pyplot as plt


import pylab as pl

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as KMeansGood
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import silhouette_score


def TR(row):
    TR = max([(row["high"] - row["low"]), abs(row["high"] - row["close_lag"]), abs(row["close_lag"] - row["low"])])
    
    return TR

def time_weighted_mean(array, decay):
    N = len(array)
    
    if N == 0:
        return 0

    if decay != 1:
        mean = ((1-decay)/(1-decay**N)) * sum([(decay**(N-i-1) * array.iloc[i]) for i in range(N)]) 

    else:
        mean = array.mean()

    return mean


def Annual_capital(data):
    year = data['year'].unique().tolist()
    capital_list = []
    for y in year:
        df = data[data['year'] == y]
        capital = df[df['capital']!=0]['capital'].mean()
        capital_list.append(capital)

    annual_capital = time_weighted_mean(pd.Series(capital_list), decay = 0.9)

    return annual_capital


##Compute Technical Indicators    
def get_cluster_feature(data, decay):
    data = data.reset_index(drop=True).sort_values(by='ts')
    dataset = data.copy()
       
    dataset['close_lag'] = dataset['close'].shift(1)

    # Create True Range
    dataset['TR'] = dataset.apply(TR, axis=1)
    dataset['TR_scale'] = dataset['TR']/dataset['close']
    
    price_ATR = time_weighted_mean(dataset['TR_scale'].dropna(), decay=decay)
    total_mean = time_weighted_mean(dataset[dataset['total'] != 0]['total'].dropna(), decay=decay)
    capital = Annual_capital(dataset)
    stockno = dataset['StockNo'].iloc[0]

    series = pd.Series([total_mean, price_ATR, capital]).rename(stockno)

    return series


def Normalize(data, scaler='minmax0'):
    df = data.copy()
    if scaler.lower() == 'minmax0':
        sc = MinMaxScaler(feature_range = (0, 1))
    elif scaler.lower() == 'minmax1':
        sc = MinMaxScaler(feature_range = (-1, 1))
    elif scaler.lower() == 'standardize':
        sc = StandardScaler()
    elif scaler.lower() == 'normalize':
        sc = Normalizer()
    else: 
        return 'Scaler not exist'

    df.loc[:, :] = sc.fit_transform(data)

    return df


class KMeans(BaseEstimator):

    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        self.labels_ = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).argmin(axis=1)

    def _average(self, X):
        return X.mean(axis=0)

    def _m_step(self, X):
        X_center = None
        for center_id in range(self.k):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers_[center_id] = X_center
            else:
                self.cluster_centers_[center_id] = \
                    self._average(X[center_mask])

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.k]
        self.cluster_centers_ = X[self.labels_]

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self

class KMedians(KMeans):

    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

    def _average(self, X):
        return np.median(X, axis=0)



def Cluster(data, method, cluster_list):

    '''
    Stock Clustering
    Input Correlation matrix.
    method = {Kmeans, Kmedians, Hierarchical}
    cluster_list shall be the number of clusters wanted

    Output label of each stock in different cluster numbers; metrics
    '''

    X = data.values

    sc_scores = []
    label_list = []
    model = []

    for i, c in enumerate(tqdm(cluster_list, total=len(cluster_list))):
        if method == 'Kmeans':
            cluster_model = KMeansGood(n_clusters=c, init='k-means++', max_iter=1000).fit(X)
        
        elif method == 'Kmedians':
            cluster_model = KMedians(k=c, max_iter=1000)
            cluster_model.fit(X)

        else:
            cluster_model = AgglomerativeClustering(n_clusters=c, affinity='euclidean', linkage='ward')
            cluster_model.fit_predict(X)

        label = cluster_model.labels_.tolist()
        label_list.append(label)


        sc_score = silhouette_score(X, cluster_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)

        model.append(cluster_model)

    label_df = pd.DataFrame(np.stack(label_list, axis=1), columns=[str(i) for i in cluster_list], index=data.index)

    return label_df, sc_scores, model


def ClusterStatistics(cluster_data, origin_df, cluster_num):

    '''
    Input a dataframe of stock labels, correlation matrix, and the total number of clusters.
    Return a matrix which the elements represents the correlation mean of stocks in the two clusters.
    '''

    ClusterNo = str(cluster_num)
    mean_list = []
    std_list = []

    for i in range(cluster_num):
        stock_list = cluster_data[cluster_data[ClusterNo] == i].index.tolist()
        d = origin_df.loc[stock_list]
        mean, std = d.mean().rename(f'Cluster{i}'), d.std().rename(f'Cluster{i}')
        mean_list.append(mean)
        std_list.append(std)

    mean_df = pd.concat(mean_list, axis=1).transpose()
    std_df = pd.concat(std_list, axis=1).transpose()

    return mean_df, std_df


def FindCluster(stock, all_df, centroid):

    '''
    Find the cluster a stock not in any cluster belongs to by computing the mean correlation with other stocks.
    Input the stock data wanted, the data containing every stocks, and the dataframe with all labels.
    Return the cluster a stock belongs to. 
    '''
    stock_array = all_df.loc[stock]

    distance = [np.linalg.norm(stock_array - np.array(c)) for c in centroid]
    cluster_index = distance.index(min(distance))


    return [stock, cluster_index]
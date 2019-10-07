import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta, date
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import MinMaxScaler
import pywt
import copy
import matplotlib.pyplot as plt


import pylab as pl

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeansGood
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import silhouette_score


def WT(index_list, wavefunc='db4', lv=4, m=1, n=4, plot=False):
    
    '''
    WT: Wavelet Transformation Function
    index_list: Input Sequence;
   
    lv: Decomposing Level；
 
    wavefunc: Function of Wavelet, 'db4' default；
    
    m, n: Level of Threshold Processing
   
    '''
   
    # Decomposing 
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function 

    # Denoising
    # Soft Threshold Processing Method
    for i in range(m,n+1):   #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0   # Set to zero if smaller than threshold

    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])
    
    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]
            
    denoised_index = np.sum(coeff, axis=0)   
        
    if plot:     
        data.plot(figsize=(10,5))
        plt.title(f'Level_{lv}')
   
    return denoised_index


def Close_denoise(data, feature='close', levels=6, scale=True):

    '''
    Input data and the feature wanted to denoise.
    If scale = False, will not scale the feature to range(0,1)
    Higher level captures longer trend (Not so Accurate)

    '''
    
    d = data.copy().sort_values(by='ts').reset_index(drop=True)
    if scale:
        sc = MinMaxScaler(feature_range = (0, 1))
        a = sc.fit_transform(np.array(d[feature]).reshape(len(d), 1))
        d['close_normalized'] = a
    
        d['denoised'] =  WT(d['close_normalized'], lv=levels, n=levels)

    else:
        d['denoised'] =  WT(d['close'], lv=levels, n=levels)

    return d


def Cluster_df(data, End, Start=date(2004,2,11), feature='denoised'):

    '''
    Return a dataframe ready for correlation computation.
    The output is a dataframe with multiple time series.
    Shall select the features wanted to use for correlation computation.

    '''

    dates = [Start + timedelta(days=x) for x in range((End-Start).days)]
    
    df = data[data.ts.isin(dates)]
    df_list = [group[1] for group in df.groupby(df['StockNo'])]

    return_list = []
    for i, d in enumerate(tqdm(df_list, total=len(df_list))):
        
        
        Stockname = d['StockName'].unique()[0]
        d = d.sort_values(by='ts').set_index('ts')

        Series = (d[feature]).rename(Stockname)
        return_list.append(Series.iloc[1:])

    Return_df = pd.concat(return_list, axis=1)

    return Return_df




class CorrClass():
    def __init__(self):
        self.corr_dict = {}
        print("i am someone u did not know.")
    
    def correlation(pars):
        s1 = max(stock, stock_other)
        s2 = min(stock, stock_other)        
        if (s1, s2) in self.corr_dict:
             x = self.corr_dict[(s1, s2)]
        else:
            pass


cc = CorrClass()
cc.correlation()
    


def Correlation(stock, data, all_stocks, corr_dict):

    '''
    Return the correlation list of a given stock with all other stocks in dataframe.
    '''
    
    for stock_other in all_stocks:
        df = data[[stock, stock_other]].dropna()
    corr_list = [data[[stock, stock_other]].dropna().iloc[:, 0].corr(data[[stock, stock_other]].dropna().iloc[:, 1]) for stock_other in all_stocks]
    
    corr = pd.Series(corr_list).rename(stock)
        
    return corr




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


def ClusterCorrMatrix(cluster_data, corr_matrix, cluster_num):

    '''
    Input a dataframe of stock labels, correlation matrix, and the total number of clusters.
    Return a matrix which the elements represents the correlation mean of stocks in the two clusters.
    '''

    ClusterNo = str(cluster_num)
    cluster_corr = pd.DataFrame(index=['Cluster_{}'.format(i) for i in range(1, cluster_num+1)], columns=['Cluster_{}'.format(i) for i in range(1, cluster_num+1)])

    for i in range(cluster_num):
        d_i = cluster_data[cluster_data[ClusterNo] == i]
        for j in range(i, cluster_num):
            d_j = cluster_data[cluster_data[ClusterNo] == j]
            cluster_corr.iloc[i, j], cluster_corr.iloc[j, i] = corr_matrix.loc[d_i.index.tolist(), d_j.index.tolist()].mean().mean(), corr_matrix.loc[d_i.index.tolist(), d_j.index.tolist()].mean().mean()

    
    return cluster_corr


def FindCluster(stock_name, all_df, cluster_df, cluster_num=9):

    '''
    Find the cluster a stock not in any cluster belongs to by computing the mean correlation with other stocks.
    Input the stock data wanted, the data containing every stocks, and the dataframe with all labels.
    Return the cluster a stock belongs to. 
    '''
    cluster_total = str(cluster_num)
    cluster_list = [cluster_df[cluster_df[cluster_total] == i].index.tolist() for i in range(cluster_num)]
    corr_list = []


    for stock_list in cluster_list:
        corr_mean = np.array([all_df[[stock_name, stock]].dropna().iloc[:, 0].corr(all_df[[stock_name, stock]].dropna().iloc[:, 1]) for stock in stock_list]).mean()
        corr_list.append(corr_mean)

    cluster_num = corr_list.index(max(corr_list))

    return [stock_name, cluster_num]




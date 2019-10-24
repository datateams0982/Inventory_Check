import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from Model_class import CNN_model, CNN_Tree_Classifier, CNN_Bagging, CNN_Boosting
from Fine_Tune import Hyperparameter_Tuning as Tuning
import time
import pickle
import random

def main():
    path = 'D:\\庫存健診開發\\data\\Training\\processed\\'
    CNN_param = {'Conv_layer': [2, 3],
                'Dense_layer': [1, 2],
                'Dense': [64, 128, 256],
                'dropout_ratio': [0, 0.1, 0.15, 0.2, 0.25, 0.3],
                'filter': [64, 100, 128, 256],
                'kernel': [3, 5],
                'padding': ['causal'],
                'learning_rate': [0.001, 0.002, 0.003],
                'batch': [64, 128, 256, 512],
                'epochs': [30, 40, 50, 70, 80]}
    model_param = {'n_estimator': [5, 10],
                    'subsample': [0.75, 0.8, 0.85, 0.9]}

    Start = time.time()
    Optimization = Tuning(df_path=path, model='CNN_Boosting', CNNparam=CNN_param, Modelparam=model_param, max_iter=15, cluster_num=0)
    Optimization.RandomSearch()

    end = time.time()

    total = end - Start

    with open('D:\\庫存健診開發\\CNN_Boosting.txt', 'wb') as fp:
        fp.write(total)



if __name__ == "__main__":
	main()
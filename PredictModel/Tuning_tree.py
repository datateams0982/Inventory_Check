import numpy as np 
import pandas as pd


from Model_class import CNN_model, CNN_Tree_Classifier, CNN_Bagging, CNN_Boosting
from Fine_Tune import Hyperparameter_Tuning as Tuning

import pickle
import random

import time

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
                'epochs': [30, 40, 50, 60]}
    rf_param = {'classifier': ['RandomForest'],
                'max_depth': [10, 15, 20, 25, 30],
                'max_features': ['auto'],
                'min_samples_leaf': [0.05, 0.1, 0.15, 0.2],
                'min_samples_split': [0.05, 0.1, 0.15, 0.2],
                'n_estimators': [50, 75, 100, 125, 150]}

    start = time.time()
    Optimization = Tuning(df_path=path, model='CNN_Tree_Classifier', CNNparam=CNN_param, Modelparam=rf_param, max_iter=30, cluster_num=0)
    Optimization.RandomSearch()
    end = time.time()

    total = end - start

    with open('D:\\庫存健診開發\\CNN_tree_tuning_time.txt', 'wb') as fp:
        fp.write(total)



if __name__ == "__main__":
	main()
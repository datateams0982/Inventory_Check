import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from Model_class import CNN_model, CNN_Tree_Classifier, CNN_Bagging, CNN_Boosting
from Fine_Tune import Hyperparameter_Tuning as Tuning

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
    model_param = {'n_estimator': [25, 50, 75, 100, 125],
                    'subsample': [0.75, 0.8, 0.85, 0.9]}

    Optimization = Tuning(df_path=path, model='CNN_Bagging', CNNparam=CNN_param, Modelparam=rf_param, max_iter=1000, cluster_num=0, model_keep=3)
    Optimization.RandomSearch()
    Optimization.save_optimal_model()
    Optimization.save_optimal_param()


if __name__ == "__main__":
	main()
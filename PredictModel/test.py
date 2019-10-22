import numpy as np # linear algebra
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
# tqdm.pandas()

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from functools import partial


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, TimeDistributed, LeakyReLU, Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from xgboost import XGBClassifier, plot_importance
from multiprocessing import Pool

from Model_class import CNN_model, CNN_Tree_Classifier, CNN_Bagging, CNN_Boosting
import Model_class as func

import pickle
import random


if __name__ == '__main__':
    cluster = 0
    with open(f'D:\\庫存健診開發\\data\\Training\\processed\\Cluster_{cluster}_classification_minmax0_Weekly', 'rb') as fp:
        load_list = pickle.load(fp)

    param = {'Conv_layer': 3,
                                'Dense_layer': 1,
                                'Dense': [128],
                                'dropout_ratio': [0.2, 0.2, 0.2],
                                'filter': [64, 128, 128],
                                'kernel': [3, 3, 3],
                                'padding': ['causal', 'causal', 'causal'],
                                'learning_rate': 0.001,
                                'batch': 256,
                                'epochs': 1}
                                
    classifier = CNN_Tree_Classifier(load_list, classifier='xgboost', CNNparam=param)
    classifier.get_dependent_variable()

    # print(type(classifier._training_feature))


    argument = [classifier._X_train, classifier._y_train, classifier._X_val, classifier._y_val, classifier._X_test, classifier._CNNparam, 'D:\\庫存健診開發\\model\\test_']
    with Pool(1) as p:
        training_feature, val_feature, testing_feature = p.apply(func.CNN_train, (argument, ))

    with open('D:\\庫存健診開發\\model\\train_400', 'wb') as fp:
        pickle.dump(training_feature[:, :400], fp)

    with open('D:\\庫存健診開發\\model\\train_2560', 'wb') as fp:
        pickle.dump(training_feature, fp)

    with open('D:\\庫存健診開發\\model\\val_400', 'wb') as fp:
        pickle.dump(val_feature[:, :400], fp)

    with open('D:\\庫存健診開發\\model\\val_2560', 'wb') as fp:
        pickle.dump(val_feature, fp)

    with open('D:\\庫存健診開發\\model\\test_400', 'wb') as fp:
        pickle.dump(testing_feature[:, :400], fp)

    with open('D:\\庫存健診開發\\model\\test_2560', 'wb') as fp:
        pickle.dump(testing_feature, fp)


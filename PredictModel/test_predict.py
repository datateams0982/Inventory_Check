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

cluster = 0
with open(f'D:\\庫存健診開發\\data\\Training\\processed\\Cluster_{cluster}_classification_minmax0_Weekly', 'rb') as fp:
    load_list = pickle.load(fp)

# with open('D:\\庫存健診開發\\model\\train_400', 'rb') as fp:
#     training_feature_short = pickle.load(fp)

with open('D:\\庫存健診開發\\model\\train_2560', 'rb') as fp:
    training_feature = pickle.load(fp)

# with open('D:\\庫存健診開發\\model\\val_400', 'rb') as fp:
#     val_feature_short = pickle.load(fp)

with open('D:\\庫存健診開發\\model\\val_2560', 'rb') as fp:
    val_feature = pickle.load(fp)

# with open('D:\\庫存健診開發\\model\\test_400', 'rb') as fp:
#     testing_feature_short = pickle.load(fp)

with open('D:\\庫存健診開發\\model\\test_2560', 'rb') as fp:
    testing_feature = pickle.load(fp)


xgbc_param={'colsample_bytree': 0.8,
                                        'subsample': 0.8,
                                        'n_estimators': 10, 
                                        'max_depth': 6, 
                                        'gamma': 0.01,
                                        'eta': 0.1}
                            
classifier = CNN_Tree_Classifier(load_list, classifier='xgboost', xgbc_param=xgbc_param)
classifier.get_dependent_variable()
classifier._training_feature = training_feature
classifier._val_feature = val_feature
classifier._testing_feature = testing_feature


print('Reload model')
with open('D:\\庫存健診開發\\model\\test_classifier', 'rb') as fp:
    classifier._classifier = joblib.load(fp)

prediction = classifier.predict(target='test')
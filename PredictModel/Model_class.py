import numpy as np # linear algebra
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import random
# tqdm.pandas()

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, TimeDistributed, LeakyReLU, Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from xgboost import XGBClassifier, plot_importance

import pickle
import Model_function as func


class CNN_model:

    def __init__(self, df, CNNparam={'Conv_layer': 3,
                        'dropout': True,
                        'Dense': True,
                        'dropout_ratio': 0.2,
                        'l1_filter': 64,
                        'l2_filter': 128,
                        'l3_filter': 128,
                        'l1_kernel': 3,
                        'l2_kernel': 3,
                        'l3_kernel': 3,
                        'padding1': 'causal',
                        'padding2': 'causal',
                        'padding3': 'causal',
                        'learning_rate': 0.001,
                        'batch': 256,
                        'epochs': 80}):
        self._X_train = np.array(df[0])
        self._Y_train = df[1]
        self._X_val = np.array(df[2])
        self._Y_val = df[3]
        self._X_test = np.array(df[4])
        self._Y_test = df[5]
        self._CNNparam = CNNparam

        self._CNN = Sequential()
        self._CNN.add(Conv1D(
                filters=self._CNNparam['l1_filter'], 
                kernel_size=self._CNNparam['l1_kernel'], 
                activation='relu', 
                input_shape=(self._X_train.shape[1], self._X_train.shape[2]), 
                padding=self._CNNparam['padding1']))
        
        self._CNN.add(Conv1D(
                filters=self._CNNparam['l2_filter'], 
                kernel_size=self._CNNparam['l2_kernel'], 
                activation='relu',
                padding=self._CNNparam['padding2']))
            
        if (self._CNNparam['dropout']) and (self._CNNparam['Conv_layer'] <= 2):
            self._CNN.add(Dropout(self._CNNparam['dropout_ratio']))
        
        if self._CNNparam['Conv_layer'] > 2:
            self._CNN.add(Conv1D(
                filters=self._CNNparam['l3_filter'], 
                kernel_size=self._CNNparam['l3_kernel'], 
                activation='relu',
                padding=self._CNNparam['padding3']))
            if self._CNNparam['dropout']:
                self._CNN.add(Dropout(self._CNNparam['dropout_ratio']))
                
        self._CNN.add(Flatten(name='feature'))
        
        if self._CNNparam['Dense']:
            self._CNN.add(Dense(self._CNNparam['Dense']))
            self._CNN.add(Dropout(self._CNNparam['dropout_ratio']))
            
        self._CNN.add(Dense(2, activation='softmax'))

        self._opt = optimizers.Nadam(lr=self._CNNparam['learning_rate'])
        self._CNN.compile(loss='categorical_crossentropy', optimizer=self._opt, metrics=[tf.keras.metrics.AUC()])
        
    
    
    @property
    def X_train(self):
        return self._X_train.shape
    
    @property
    def X_val(self):
        return self._X_val.shape
    
    @property
    def X_test(self):
        return self._X_test.shape
    
    @property
    def Y_train(self):
        return self._Y_train
    
    @property
    def Y_val(self):
        return self._Y_val
    
    @property
    def Y_test(self):
        return self._Y_test
    
    def _check_status(self):
        if (len(self._X_train) != len(self._Y_train)) or (len(self._X_val) != len(self._Y_val)) or (len(self._X_test) != len(self._Y_test)):
            raise('Invalid Data Shape')

    
    def _categorical_transform(self, data):

        onehot_encoder = OneHotEncoder(sparse=False)

        d = np.array(data).reshape(len(data), 1)
        Y = onehot_encoder.fit_transform(d)

        return Y
    
        
    def get_dependent_variable(self):
        
        try:
            self._check_status()
        except Exception:
            print('Invalid data shape. Check data size.')
            
            
        self._lab_train = np.array([item[2] for item in self.Y_train])
        self._lab_val = np.array([item[2] for item in self.Y_val])
        self._lab_test = np.array([item[2] for item in self.Y_test])
        
        self._y_train = self._categorical_transform(self._lab_train)
        self._y_val = self._categorical_transform(self._lab_val)
        self._y_test = self._categorical_transform(self._lab_test)

    
    @property
    def CNNparam(self):
        return self._CNNparam
        

    def CNN_train(self):
        
        self._CNN.fit(self._X_train, self._y_train, validation_data=(self._X_val, self._y_val), batch_size=self._CNNparam['batch'], epochs=self._CNNparam['epochs'], verbose=2)

    
    @property    
    def CNN(self):
        
        return self._CNN.summary()

        
    def load_CNN(self, model_path):

        self._CNN = tf.keras.models.load_model(model_path)  
        
    def save_CNN(self, path):
        
        self._CNN.save(f'{path}.h5')
        
    def get_layer_weight(self):
        
        weight_list = []
        for layer in self._CNN.layers:
            weight_list.append(layer.get_weights())
            
        return weight_list
    
    def Evaluation(self):
        
        score = self._CNN.evaluate(self._X_test, self._y_test, batch_size=4096)
        
        prediction = self._CNN.predict(self._X_test)
        y_pred = [item.index(max(item)) for item in prediction.tolist()]
        
        target_names = ['down', 'up']
        report = classification_report(self._lab_test.tolist(), y_pred, target_names=target_names)
        
        return [score, report]
    
    def prediction(self, target):
        
        if target == 'train':
            prediction = self._CNN.predict(self._X_train)
        elif target == 'validation':
            prediction = self._CNN.predict(self._X_val)
        else:
            prediction = self._CNN.predict(self._X_test)
            
        y_pred = [item.index(max(item)) for item in prediction.tolist()]
            
        return y_pred
    
    def Separate_Evaluation(self, target, **kwargs):
        
        if target == 'train':
            prediction = self._CNN.predict(self._X_train)
            Real = self._Y_train
            
        elif target == 'validation':
            prediction = self._CNN.predict(self._X_val)
            Real = self._Y_val
            
        else:
            prediction = self._CNN.predict(self._X_test)
            Real = self._Y_test
            
        y_pred = [item.index(max(item)) for item in prediction.tolist()]

        
        if len(kwargs.keys()) == 1:
            if 'year' in kwargs:
                value = kwargs['year']
                y_true = [sublist[2] for sublist in Real if sublist[0].year == value]  
                y_index = [i for i, sublist in enumerate(Real) if sublist[0].year == value]
            else:
                value = kwargs['stock']
                y_true = [sublist[2] for sublist in Real if sublist[1] == value]  
                y_index = [i for i, sublist in enumerate(Real) if sublist[1] == value]

        else:
            year = kwargs['year']
            stock = kwargs['stock']
            y_true = [sublist[2] for sublist in Real if (sublist[0].year == year) and (sublist[1] == stock)]  
            y_index = [i for i, sublist in enumerate(Real) if (sublist[0].year == year) and (sublist[1] == stock)]

        y = [y_pred[i] for i in y_index]
        

                
        target_names = ['down', 'up']
        
        report = classification_report(y_true, y, target_names=target_names)
        accuracy = metrics.accuracy_score(y_true, y)
        

        return report, accuracy


class CNN_Tree_Classifier(CNN_model):

    def __init__(self, df, classifier, 
                        CNNparam={'Conv_layer': 3,
                                        'dropout': True,
                                        'Dense': True,
                                        'dropout_ratio': 0.2,
                                        'l1_filter': 64,
                                        'l2_filter': 128,
                                        'l3_filter': 128,
                                        'l1_kernel': 3,
                                        'l2_kernel': 3,
                                        'l3_kernel': 3,
                                        'padding1': 'causal',
                                        'padding2': 'causal',
                                        'padding3': 'causal',
                                        'learning_rate': 0.001,
                                        'batch': 256,
                                        'epochs': 80}, 
                        rf_param={'max_depth': 10,
                                        'max_features': 'auto',
                                        'min_samples_leaf': 0.1,
                                        'min_samples_split': 0.1,
                                        'n_estimators': 100},
                        xgbc_param={'colsample_bytree': 0.8,
                                        'subsample': 0.8,
                                        'n_estimators': 100, 
                                        'max_depth': 6, 
                                        'gamma': 0.01,
                                        'eta': 0.1}):
        super().__init__(df, CNNparam)
        
        self.type = classifier.lower()
        
        if classifier.lower() == 'randomforest':
            self._Classifierparam = rf_param
                                        
            self._classifier = RandomForestClassifier(n_estimators=self._Classifierparam['n_estimators'], 
                                                        max_depth=self._Classifierparam['n_estimators'],
                                                        max_features=self._Classifierparam['max_features'],
                                                        min_samples_leaf=self._Classifierparam['min_samples_leaf'], 
                                                        min_samples_split=self._Classifierparam['min_samples_split'],
                                                        verbose=1, 
                                                        n_jobs=-1)

        elif classifier.lower() == 'xgboost':
            self._Classifierparam = xgbc_param

            self._classifier = XGBClassifier(colsample_bytree=self._Classifierparam['colsample_bytree'],
                                                subsample=self._Classifierparam['subsample'],
                                                n_estimators=self._Classifierparam['n_estimators'], 
                                                max_depth=self._Classifierparam['n_estimators'], 
                                                gamma=self._Classifierparam['gamma'],
                                                eta=self._Classifierparam['eta'],
                                                updater='grow_gpu_hist')
            
    
            
    def Feature_extraction(self):
        intermediate_layer_model = Model(inputs=self._CNN.input, outputs=self._CNN.get_layer('feature').output)
        self._training_feature = intermediate_layer_model.predict(self._X_train)
        self._val_feature = intermediate_layer_model.predict(self._X_val)
        self._testing_feature = intermediate_layer_model.predict(self._X_test)
        
    @property    
    def training_feature(self):
        return self._training_feature
    
    @property    
    def val_feature(self):
        return self._val_feature
    
    @property    
    def testing_feature(self):
        return self._testing_feature

    @property
    def Classifierparam(self):
        return self._Classifierparam
        
        
    def Classification(self):
        
        if self.type == 'randomforest':
            self._classifier.fit(self._training_feature, self._lab_train)
        elif self.type == 'xgboost':
            self._classifier.fit(self._training_feature_, self._lab_train, eval_set = [(self._training_feature_, self._lab_train), (self._val_feature_, self._lab_val)], verbose=True, eval_metric='auc')

    def save_model(self, path):
        
        with open(f'{path}', 'wb') as fp:
            joblib.dump(self._classifier, fp) 
            
    def load_model(self, model_path):
        
        with open(model_path, 'rb') as fp:
            self._classifier = joblib.load(fp)

    def Evaluation(self):
        
        prediction = self._classifier.predict(self._testing_feature)
        
        fpr, tpr, thresholds = metrics.roc_curve(self._lab_test, prediction)
        auc = metrics.auc(fpr, tpr)
        accuracy = metrics.accuracy_score(self._lab_test, prediction)
        
        target_names = ['down', 'up']
        report = classification_report(self._lab_test.tolist(), prediction.tolist(), target_names=target_names)
        
        return [auc, accuracy, report]
    
    
    def prediction(self, target):
        
        if target == 'train':
            prediction = self._classifier.predict(self._training_feature)
        elif target == 'validation':
            prediction = self._classifier.predict(self._val_feature)
        else:
            prediction = self._classifier.predict(self._testing_feature)
            
        return prediction
    
    
    def Separate_Evaluation(self, target, **kwargs):
        
        if target == 'train':
            prediction = self._classifier.predict(self._training_feature)
            Real = self._Y_train
            
        elif target == 'validation':
            prediction = self._classifier.predict(self._val_feature)
            Real = self._Y_val
            
        else:
            prediction = self._classifier.predict(self._testing_feature)
            Real = self._Y_test
            
        y_pred = prediction.tolist()
        
        if len(kwargs.keys()) == 1:
            if 'year' in kwargs:
                value = kwargs['year']
                y_true = [sublist[2] for sublist in Real if sublist[0].year == value]  
                y_index = [i for i, sublist in enumerate(Real) if sublist[0].year == value]
            else:
                value = kwargs['stock']
                y_true = [sublist[2] for sublist in Real if sublist[1] == value]  
                y_index = [i for i, sublist in enumerate(Real) if sublist[1] == value]

        else:
            year = kwargs['year']
            stock = kwargs['stock']
            y_true = [sublist[2] for sublist in Real if (sublist[0].year == year) and (sublist[1] == stock)]  
            y_index = [i for i, sublist in enumerate(Real) if (sublist[0].year == year) and (sublist[1] == stock)]

        y = [y_pred[i] for i in y_index]
        
        
                
        target_names = ['down', 'up']
        
        report = classification_report(y_true, y, target_names=target_names)
        accuracy = metrics.accuracy_score(y_true, y)
        

        return report, accuracy
     

class CNN_Bagging(CNN_model):

    def __init__(self, df, verbose=1,
                        Param={'n_estimator': 100,
                                'col_sample': 1.0,
                                'subsample': 0.8},
                        CNNparam={'Conv_layer': 3,
                                    'dropout': True,
                                    'Dense': True,
                                    'dropout_ratio': 0.2,
                                    'l1_filter': 64,
                                    'l2_filter': 128,
                                    'l3_filter': 128,
                                    'l1_kernel': 3,
                                    'l2_kernel': 3,
                                    'l3_kernel': 3,
                                    'padding1': 'causal',
                                    'padding2': 'causal',
                                    'padding3': 'causal',
                                    'learning_rate': 0.001,
                                    'batch': 256,
                                    'epochs': 80}):

        super().__init__(df, CNNparam)
        self.n_estimator = Param['n_estimator']
        self.col_sample = Param['col_sample']
        self.subsample = Param['subsample']


    def Bootstrap(self):

        total = len(self._X_train)
        sample_size = 
        index = [random.randint(0, total-1) for i in range(0, )]

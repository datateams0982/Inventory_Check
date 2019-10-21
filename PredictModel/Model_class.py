import numpy as np # linear algebra
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import random
import os
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


class CNN_model:

    def __init__(self, df, CNNparam={'Conv_layer': 3,
                        'Dense_layer': 1,
                        'Dense': [128],
                        'dropout_ratio': [0.2, 0.2, 0.2],
                        'filter': [64, 128, 128],
                        'kernel': [3, 3, 3],
                        'padding': ['causal', 'causal', 'causal'],
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
                filters=self._CNNparam['filter'][0], 
                kernel_size=self._CNNparam['kernel'][0], 
                activation='relu', 
                input_shape=(self._X_train.shape[1], self._X_train.shape[2]), 
                padding=self._CNNparam['padding'][0]))

        for i in range(self._CNNparam['Conv_layer'] - 1):
            self._CNN.add(Conv1D(
                filters=self._CNNparam['filter'][i+1], 
                kernel_size=self._CNNparam['kernel'][i+1], 
                activation='relu',
                padding=self._CNNparam['padding'][i+1]))

            self._CNN.add(Dropout(self._CNNparam['dropout_ratio'][i]))
            
                
        self._CNN.add(Flatten(name='feature'))
        
        for i in range(self._CNNparam['Dense_layer'] - 1):
            self._CNN.add(Dense(self._CNNparam['Dense'][i]))
            self._CNN.add(Dropout(self._CNNparam['dropout_ratio'][i+self._CNNparam['Conv_layer'] - 1]))


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
    
    def Evaluation(self, target):
        
        prediction = self.predict(target)

        if target == 'train':
            Y = self._lab_train
        elif target == 'validation':
            Y = self._lab_val
        else:
            Y = self._lab_test

        fpr, tpr, thresholds = metrics.roc_curve(Y, prediction)
        auc = metrics.auc(fpr, tpr)
        accuracy = metrics.accuracy_score(Y, prediction)
        
        target_names = ['down', 'up']
        report = classification_report(Y.tolist(), prediction.tolist(), target_names=target_names)
        
        return [auc, accuracy, report]
    
    def predict(self, target):
        
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
            prediction = self.predict(target)
            Real = self._Y_train
            
        elif target == 'validation':
            prediction = self.predict(target)
            Real = self._Y_val
            
        else:
            prediction = self.predict(target)
            Real = self._Y_test

        
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

    def Overall_Evaluation(self, target, threshold=0.6):

        if target  == 'train':
            stock = [sublist[1] for sublist in self._Y_train]

        elif target == 'validation':
            stock = [sublist[1] for sublist in self._Y_val]

        else:
            stock = [sublist[1] for sublist in self._Y_test]

        acc = []


        for s in stock:
            r, a = self.Separate_Evaluation(target, stock=s)
            acc.append(a)
            


        rate = len([0 for i in acc if i < threshold])/len(acc)

        return 


class CNN_Tree_Classifier(CNN_model):

    def __init__(self, df, classifier='xgboost', 
                        CNNparam={'Conv_layer': 3,
                                        'Dense_layer': 1,
                                        'Dense': [128],
                                        'dropout_ratio': [0.2, 0.2, 0.2],
                                        'filter': [64, 128, 128],
                                        'kernel': [3, 3, 3],
                                        'padding': ['causal', 'causal', 'causal'],
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
                                                tree_method='hist',
                                                gpu_id=0)
            
    
            
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
            self._classifier.fit(self._training_feature, self._lab_train, eval_set = [(self._training_feature, self._lab_train), (self._val_feature, self._lab_val)], verbose=True, eval_metric='auc')

    def save_model(self, path):
        
        with open(f'{path}classifier', 'wb') as fp:
            joblib.dump(self._classifier, fp) 

        self._CNN.save(f'{path}CNN.h5')
        
            
    def load_model(self, path):
        
        with open(f'{path}classifier', 'rb') as fp:
            self._classifier = joblib.load(fp)

        self._CNN = tf.keras.models.load_model(f'{path}CNN.h5')
        self.Feature_extraction()
    
    def predict(self, target):
        
        if target == 'train':
            prediction = self._classifier.predict(self._training_feature)
        elif target == 'validation':
            prediction = self._classifier.predict(self._val_feature)
        else:
            prediction = self._classifier.predict(self._testing_feature)
            
        return prediction
    
    
     

class CNN_Bagging(CNN_model):

    def __init__(self, df, verbose=1,
                        Param={'n_estimator': 100,
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

        self._X_train = np.array(df[0])
        self._Y_train = df[1]
        self._X_val = np.array(df[2])
        self._Y_val = df[3]
        self._X_test = np.array(df[4])
        self._Y_test = df[5]
        self._CNNparam = CNNparam
        self.n_estimator = Param['n_estimator']
        self.subsample = Param['subsample']
        self._total = len(self._X_train)
        self._sample_size = int(self._total*self.subsample)


    def _Bootstrap(self):

        index = random.sample(range(self._total), self._sample_size)

        X_batch = self._X_train[index]
        Y_batch = self._y_train[index]

        return X_batch, Y_batch

    def CNN_train(self):

        self.model_list = []
        
        for i in tqdm(range(self.n_estimator), total=self.n_estimator):
            X, Y = self._Bootstrap()

            self._CNN = Sequential()
            self._CNN.add(Conv1D(
                filters=self._CNNparam['filter'][0], 
                kernel_size=self._CNNparam['kernel'][0], 
                activation='relu', 
                input_shape=(self._X_train.shape[1], self._X_train.shape[2]), 
                padding=self._CNNparam['padding'][0]))

            for i in range(self._CNNparam['Conv_layer'] - 1):
                self._CNN.add(Conv1D(
                    filters=self._CNNparam['filter'][i+1], 
                    kernel_size=self._CNNparam['kernel'][i+1], 
                    activation='relu',
                    padding=self._CNNparam['padding'][i+1]))

                self._CNN.add(Dropout(self._CNNparam['dropout_ratio'][i]))
            
                
            self._CNN.add(Flatten(name='feature'))
        
            for i in range(self._CNNparam['Dense_layer'] - 1):
                self._CNN.add(Dense(self._CNNparam['Dense'][i]))
                self._CNN.add(Dropout(self._CNNparam['dropout_ratio'][i+self._CNNparam['Conv_layer'] - 1]))


            self._CNN.add(Dense(2, activation='softmax'))

            self._opt = optimizers.Nadam(lr=self._CNNparam['learning_rate'])
            self._CNN.compile(loss='categorical_crossentropy', optimizer=self._opt, metrics=[tf.keras.metrics.AUC()])

            self._CNN.fit(X, Y, validation_data=(self._X_val, self._y_val), batch_size=self._CNNparam['batch'], epochs=self._CNNparam['epochs'], verbose=2)
            self.model_list.append(self._CNN)

    def load_model(self, path):

        model_list = os.listdir(path)
        self.model_list = []

        for model in model_list:
            m = tf.keras.models.load_model(path)  
            self.model_list.append(m)
        
    def save_model(self, path):
        
        for i, model in enumerate(self.model_list):
            model.save(f'{path}{i}.h5')


    def predict(self, target='test'):

        if target == 'train':
            X = self._X_train
            Y = self._Y_train

        elif target == 'validation':
            X = self._X_val
            Y = self._Y_val

        else:
            X = self._X_test
            Y = self._Y_test

        self.prediction_list = []
        for model in self.model_list:
            p = model.predict(X)
            pred = [item.index(max(item)) for item in p.tolist()]
            self.prediction_list.append(pred)

        vote = sum(np.array(self.prediction_list))
        threshold = int(self.n_estimator/2)
        prediction = np.array([1 if v > threshold else 0 for v in vote])

        return prediction




class CNN_Boosting(CNN_model):

    def __init__(self, df, verbose=1,
                        Param={'n_estimator': 100,
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

        self._X_train = np.array(df[0])
        self._Y_train = df[1]
        self._X_val = np.array(df[2])
        self._Y_val = df[3]
        self._X_test = np.array(df[4])
        self._Y_test = df[5]
        self._CNNparam = CNNparam
        self.n_estimator = Param['n_estimator']
        self.subsample = Param['subsample']
        self._total = len(self._X_train)
        self._sample_size = int(self._total*self.subsample)
        self._weight = np.array([1/self._total for i in range(self._total)])


    def _Boosting(self):

        self._weight = self._weight / self._weight.sum(dtype=np.float64)

        select_index = np.random.choice(self._total, self._sample_size, p=self._weight, replace=False).tolist()

        X = self._X_train[select_index]
        Y = self._y_train[select_index]

        return X, Y

    def _reweight(self):

        score = self._CNN.evaluate(self._X_train, self._y_train, batch_size=4080)
        error = 1 - score[0]
        pred = self._CNN.predict(self._X_train)
        prediction = [item.index(max(item)) for item in pred.tolist()]
        weight = ((1 - error)/error) ** (1/2)
        new_weight = np.array([self._weight[i] * weight if prediction[i] != self._lab_train[i] else self._weight[i] / weight for i in range(self._total)])

        return weight, new_weight


    def CNN_train(self):
        
        self.model_list = []
        self._vote_weight = []

        for i in tqdm(range(self.n_estimator), total=self.n_estimator):
            X, Y = self._Boosting()

            self._CNN = Sequential()
            self._CNN.add(Conv1D(
                filters=self._CNNparam['filter'][0], 
                kernel_size=self._CNNparam['kernel'][0], 
                activation='relu', 
                input_shape=(self._X_train.shape[1], self._X_train.shape[2]), 
                padding=self._CNNparam['padding'][0]))

            for i in range(self._CNNparam['Conv_layer'] - 1):
                self._CNN.add(Conv1D(
                    filters=self._CNNparam['filter'][i+1], 
                    kernel_size=self._CNNparam['kernel'][i+1], 
                    activation='relu',
                    padding=self._CNNparam['padding'][i+1]))

                self._CNN.add(Dropout(self._CNNparam['dropout_ratio'][i]))
            
                
            self._CNN.add(Flatten(name='feature'))
        
            for i in range(self._CNNparam['Dense_layer'] - 1):
                self._CNN.add(Dense(self._CNNparam['Dense'][i]))
                self._CNN.add(Dropout(self._CNNparam['dropout_ratio'][i+self._CNNparam['Conv_layer'] - 1]))


            self._CNN.add(Dense(2, activation='softmax'))

            self._opt = optimizers.Nadam(lr=self._CNNparam['learning_rate'])
            self._CNN.compile(loss='categorical_crossentropy', optimizer=self._opt, metrics=['acc'])

            self._CNN.fit(X, Y, validation_data=(self._X_val, self._y_val), batch_size=self._CNNparam['batch'], epochs=self._CNNparam['epochs'], verbose=2)
            self.model_list.append(self._CNN)
            weight, self._weight = self._reweight()
            self._vote_weight.append(weight)

            

    def load_model(self, model_path, weight_path):

        model_list = os.listdir(model_path)
        self.model_list = []

        for model in model_list:
            m = tf.keras.models.load_model(model_path)  
            self.model_list.append(m)

        with open(weight_path, 'rb') as fp:
            self._vote_weight = pickle.load(fp)
        
        
    def save_model(self, model_path, weight_path):
        
        for i, model in enumerate(self.model_list):
            model.save(f'{model_path}{i}.h5')
        
        with open(weight_path, 'wb') as fp:
            pickle.dump(self._vote_weight, fp)


    def predict(self, target='test'):

        if target == 'train':
            X = self._X_train

        elif target == 'validation':
            X = self._X_val

        else:
            X = self._X_test

        self.prediction_list = []
        for model in self.model_list:
            p = model.predict(X)
            pred = [item.index(max(item)) for item in p.tolist()]
            self.prediction_list.append(pred)

        vote = sum([item * rate for item, rate in zip(np.array(self.prediction_list), np.array(self._vote_weight))])/sum(np.array(self._vote_weight))
        threshold = 0.5
        prediction = np.array([1 if v > threshold else 0 for v in vote])

        return prediction


    
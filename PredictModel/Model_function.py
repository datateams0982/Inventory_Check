import numpy as np # linear algebra
import pandas as pd
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import random
import os
import pickle
# tqdm.pandas()

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from Model_class import CNN_model, CNN_Tree_Classifier, CNN_Bagging, CNN_Boosting

class Hyperparameter_Tuning:
    def __init__(self, df, model, CNNparam, Modelparam, max_iter):

        self._df = df
        self._CNNparam = CNNparam
        self._Modelparam = Modelparam
        self._model = model.lower()
        self._iter = max_iter

    @property
    def CNNparam(self):
        return self._CNNparam
    
    @property
    def Modelparam(self):
        return self._Modelparam
    
    @property
    def iter(self):
        return self._iter

    def _Objective(self):

        if self._model == 'cnn_tree_classifier':
            classifier = CNN_Tree_Classifier(self._df, classifier=self.model_hyperparameters['classifier'], xgbc_param=self.model_hyperparameters, rf_param=self.model_hyperparameters, CNNparam=self.CNN_hyperparameters)
            classifier.get_dependent_variable()
            classifier.CNN_train()
            classifier.Feature_extraction()
            classifier.Classification()

        elif self._model == 'cnn_bagging':
            classifier = CNN_Bagging(self._df, Param=self.model_hyperparameters, CNNparam=self.CNN_hyperparameters)
            classifier.get_dependent_variable()
            classifier.CNN_train()

        else:
            classifier = CNN_Boosting(self._df, Param=self.model_hyperparameters, CNNparam=self.CNN_hyperparameters)
            classifier.get_dependent_variable()
            classifier.CNN_train()


        auc, acc, report = classifier.Evaluation(target='validation')
        balance = classifier.Overall_Evaluation(target='validation')

        print("AUC:\t{0} \t Accuracy:\t{1} \t Balance:\t{3}".format(auc, acc, balance))

        return [auc, acc, balance, self.CNN_hyperparameters, self.model_hyperparameters]


    def RandomSearch(self):
        '''Random search for hyperparameter optimization'''
    
        self.results = pd.DataFrame(columns = ['auc', 'acc', 'balance', 'CNN_params', 'Model_params'], index = list(range(self._iter)))
    
        # Keep searching until reach max evaluations
        for i in tqdm(range(self._iter), total=self._iter):
        
            # Choose random hyperparameters
            conv_layer = random.sample(self._CNNparam['Conv_layer'], 1)
            dense_layer = random.sample(self._CNNparam['Dense_layer'], 1)
            self.CNN_hyperparameters = {'Conv_layer': conv_layer,
                        'Dense_layer': dense_layer,
                        'Dense': random.sample(self._CNNparam['Dense'], dense_layer),
                        'dropout_ratio': random.sample(self._CNNparam['dropout_ratio'], conv_layer + dense_layer),
                        'filter': random.sample(self._CNNparam['filter'], conv_layer),
                        'kernel': random.sample(self._CNNparam['kernel'], conv_layer),
                        'padding': random.sample(self._CNNparam['padding'], conv_layer),
                        'learning_rate': random.sample(self._CNNparam['learning_rate'], 1),
                        'batch': random.sample(self._CNNparam['batch'], 1),
                        'epochs': random.sample(self._CNNparam['epochs'], 1)}

            self.model_hyperparameters = {k: random.sample(v, 1)[0] for k, v in self._Modelparam.items()}

            # Evaluate randomly selected hyperparameters
            eval_results = self._Objective()
        
            self.results.loc[i, :] = eval_results
        
    # Sort with best score on top
        self.results.sort_values('acc', ascending = False, inplace = True)
        self.results.reset_index(inplace = True)
        self.optimal_CNN = self.results.iloc[0]['CNN_params']
        self.optimal_model = self.results.iloc[0]['Model_params']
    
    
    def Select_Optimal(self):

        return self.results.iloc[0]

    def save_optimal_param(self, path):
        
        with open(f'{path}{self._model}_CNNparam', 'wb') as fp:
            pickle.dump(self.optiml_CNN, fp)

        with open(f'{path}{self._model}_Modelparam', 'wb') as fp:
            pickle.dump(self.optiml_model, fp)

    def load_optimal_param(self, path):

        with open(f'{path}{self._model}_CNNparam', 'rb') as fp:
            self.optimal_CNN = pickle.load(fp)

        with open(f'{path}{self._model}_Modelparam', 'rb') as fp:
            self.optimal_model = pickle.load(fp)
        

    def Train_Optimal(self):
        if self._model == 'cnn_tree_classifier':
            self.optimal = CNN_Tree_Classifier(self._df, classifier=self.optimal_model['classifier'], rf_param=self.optimal_model, CNNparam=self.optimal_CNN)
            self.optimal.get_dependent_variable()
            self.optimal.CNN_train()
            self.optimal.Feature_extraction()
            self.optimal.Classification()

        elif self._model == 'cnn_bagging':
            self.optimal = CNN_Bagging(self._df, Param=self.optimal_model, CNNparam=self.optimal_CNN)
            self.optimal.get_dependent_variable()
            self.optimal.CNN_train()

        else:
            self.optimal = CNN_Boosting(self._df, Param=self.optimal_model, CNNparam=self.optimal_CNN)
            self.optimal.get_dependent_variable()
            self.optimal.CNN_train()

        

        auc, acc, report = self.optimal.Evaluation(target='test')
        balance = self.optimal.Overall_Evaluation(target='test')
        prediction = self.optimal.predict(target='test')

        return [auc, acc, report, balance, prediction]


    def save_optimal_model(self, **kwargs):

        if self._model == 'cnn_tree_classifier':
            path = kwargs['path']
            self.optimal.save_model(path)


        elif self._model == 'cnn_bagging':
            path = kwargs['path']
            self.optimal.save_model(path)

        else:
            model_path = kwargs['model_path']
            weight_path = kwargs['weight_path']

            self.optimal.save_model(model_path, weight_path)



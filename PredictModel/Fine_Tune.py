import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import random
import os
import pickle

from Model_class import CNN_model, CNN_Tree_Classifier, CNN_Bagging, CNN_Boosting



class Hyperparameter_Tuning:

    '''
    Automated Hyperparameter Tuning
    The type of data should be [X_train, Y_train, X_val, Y_val, X_test, Y_test]
    The type of X should be a (samplesize, timestamp, feature) list
    The type of Y should be a [[year, stock, label]] list
    model should be [CNN_tree_classifier, CNN_bagging, CNN_boosting]
    The type of parameter shoud be a dictionary
    Parameter dictionary should contain all the parameters wanted to explore
    max_iter should be an integer, represents the number of iteration of random search
    cluster_num refers to the cluster now training (int)
    model_keep refers to the optimal model number wanted (int)

    '''

    def __init__(self, df_path, model, CNNparam, Modelparam, max_iter, cluster_num):
        

        assert type(max_iter) is int
        assert type(cluster_num) is int


        with open(f'{df_path}Cluster_{cluster_num}_classification_minmax0_Weekly', 'rb') as fp:
            self._df = pickle.load(fp)

        self._CNNparam = CNNparam
        self._Modelparam = Modelparam
        self._model = model.lower()

        self._iter = max_iter
        self._cluster = cluster_num
        self.optimal = []


        for i in range(1, max_iter+1):
            directory = f'D:\\庫存健診開發\\model\\cluster{cluster_num}\\{model}\\model_{i}'
            if not os.path.exists(directory):
                os.makedirs(directory)

        self._path = f'D:\\庫存健診開發\\model\\cluster{cluster_num}\\{model}\\'


    @property
    def CNNparam(self):
        return self._CNNparam
    
    @property
    def Modelparam(self):
        return self._Modelparam
    
    @property
    def iter(self):
        return self._iter


    def _Objective(self, verbose, iteration):

        '''
        The objective function (acc) to optimize in random search
        Updates optimal models and optimal accuracy
        '''

        if 'tree' in self._model:
            self.classifier = CNN_Tree_Classifier(self._df, classifier=self.model_hyperparameters['classifier'], xgbc_param=self.model_hyperparameters, rf_param=self.model_hyperparameters, CNNparam=self.CNN_hyperparameters)
            self.classifier.Classifier_train()

        elif 'bagging' in self._model:
            self.classifier = CNN_Bagging(self._df, Param=self.model_hyperparameters, CNNparam=self.CNN_hyperparameters)
            self.classifier.CNN_train()

        else:
            self.classifier = CNN_Boosting(self._df, Param=self.model_hyperparameters, CNNparam=self.CNN_hyperparameters)
            self.classifier.CNN_train()


        auc, acc, report = self.classifier.Evaluation(target='validation')
        balance_acc, balance_auc = self.classifier.Overall_Evaluation(target='validation', verbose=False)

        self.save_model(iteration=iteration)

        

        if verbose:
            print("AUC:\t{0} \t Accuracy:\t{1} \t Accuracy Balance:\t{2} \t AUC Balance:\t{3}".format(auc, acc, balance_acc, balance_auc))

        return [iteration+1, auc, acc, balance_acc, balance_auc, self.CNN_hyperparameters, self.model_hyperparameters]


    def RandomSearch(self, verbose=True):

        '''
        Random search for hyperparameter optimization
        Return Result data frame sort by accuracy score
        '''

        self.results = pd.DataFrame(columns = ['iteration', 'auc', 'acc', 'balance_acc', 'balance_auc', 'CNN_params', 'Model_params'], index = list(range(self._iter)))

        
        # Keep searching until reach max evaluations
        for i in range(self._iter):
            
            # Choose random hyperparameters
            conv_layer = random.sample(self._CNNparam['Conv_layer'], 1)[0]
            dense_layer = random.sample(self._CNNparam['Dense_layer'], 1)[0]
            self.CNN_hyperparameters = {'Conv_layer': conv_layer,
                        'Dense_layer': dense_layer,
                        'Dense': random.choices(self._CNNparam['Dense'], k=dense_layer),
                        'dropout_ratio': random.choices(self._CNNparam['dropout_ratio'], k=conv_layer + dense_layer),
                        'filter': random.choices(self._CNNparam['filter'], k=conv_layer),
                        'kernel': random.choices(self._CNNparam['kernel'], k=conv_layer),
                        'padding': random.choices(self._CNNparam['padding'], k=conv_layer),
                        'learning_rate': random.choices(self._CNNparam['learning_rate'], k=1)[0],
                        'batch': random.choices(self._CNNparam['batch'], k=1)[0],
                        'epochs': random.choices(self._CNNparam['epochs'], k=1)[0]}

            self.model_hyperparameters = {k: random.sample(v, 1)[0] for k, v in self._Modelparam.items()}

            # Evaluate randomly selected hyperparameters
            eval_results = self._Objective(verbose=verbose, iteration=i)

            self.results.loc[i, :] = eval_results
            
        
    # Sort with best score on top
        self.results.sort_values('acc', ascending = False, inplace = True)
        self.results.reset_index(drop=True)
        self.results.to_csv(f'D:\\庫存健診開發\\data\\Tuning_Result\\{self._model}.csv', index=False)
    
    

    def save_optimal_param(self):

        '''
        Save optimal parameters as pickle
        '''
        
        with open(f'{self._path}optimal_CNNparam', 'wb') as fp:
            pickle.dump(self.optimal_CNN, fp)

        with open(f'{self._path}optimal_Modelparam', 'wb') as fp:
            pickle.dump(self.optimal_model, fp)

    def load_optimal_param(self):

        with open(f'{self._path}optimal_CNNparam', 'rb') as fp:
            self.optimal_CNN = pickle.load(fp)

        with open(f'{self._path}optimal_Modelparam', 'rb') as fp:
            self.optimal_model = pickle.load(fp)
        

    # def Retrain_Optimal(self):
    #     if self._model == 'cnn_tree_classifier':
    #         self.optimal = CNN_Tree_Classifier(self._df, classifier=self.optimal_model['classifier'], rf_param=self.optimal_model, CNNparam=self.optimal_CNN)
    #         self.optimal.get_dependent_variable()
    #         self.optimal.CNN_train()
    #         self.optimal.Feature_extraction()
    #         self.optimal.Classification()

    #     elif self._model == 'cnn_bagging':
    #         self.optimal = CNN_Bagging(self._df, Param=self.optimal_model, CNNparam=self.optimal_CNN)
    #         self.optimal.get_dependent_variable()
    #         self.optimal.CNN_train()

    #     else:
    #         self.optimal = CNN_Boosting(self._df, Param=self.optimal_model, CNNparam=self.optimal_CNN)
    #         self.optimal.get_dependent_variable()
    #         self.optimal.CNN_train()

        

    #     auc, acc, report = self.optimal.Evaluation(target='test')
    #     balance = self.optimal.Overall_Evaluation(target='test')
    #     prediction = self.optimal.predict(target='test')

    #     return [auc, acc, report, balance, prediction]


    def save_model(self, iteration):

        '''
        Save optimal models
        '''
        
        if 'tree' in self._model:
            path = f'{self._path}model_{iteration+1}\\'
            self.classifier.save_model(path)

        elif 'bagging' in self._model:
            path = f'{self._path}model_{iteration+1}\\CNN_'
            self.classifier.save_model(path)

        else:
            model_path = f'{self._path}model_{iteration+1}\\CNN_'
            weight_path = f'{self._path}model_{iteration+1}\\weighting'
            self.classifier.save_model(model_path, weight_path)


    def load_optimal_model(self, iteration):

        '''
        load all optimal models
        '''
        self.results = pd.read_csv(f'D:\\庫存健診開發\\data\\Tuning_Result\\{self._model}.csv')

        if 'tree' in self._model:
            self.optimal = CNN_Tree_Classifier(self._df)
            path = f'{self._path}model_{iteration}\\'
            self.optimal.load_model(path)

        elif 'bagging' in self._model:
            self.optimal = CNN_Bagging(self._df)
            path = f'{self._path}model_{iteration}\\'
            self.optimal.load_model(path)

        else:
            self.optimal = CNN_Boosting(self._df)
            model_path = f'{self._path}model_{iteration}\\'
            weight_path = f'{self._path}model_{iteration}\\weighting'
            self.optimal.load_model(model_path, weight_path)


    def predict(self, target='test'):

        '''
        i is the numbering of optimal model wanted (int)
        target shoud be 'train', 'validation' or 'test'
        return prediction
        '''

        assert target in ['train', 'validation', 'test']
        
        prediction = self.optimal.predict(target)

        return prediction


    def Evaluation(self, target='test'):

        '''
        i is the numbering of optimal model wanted (int)
        target shoud be 'train', 'validation' or 'test'
        return auc, acc and classification report
        '''

        assert target in ['train', 'validation', 'test']
        

        score = self.optimal.Evaluation(target)

        return score

    def Separate_Evaluation(self, target='test', **kwargs):

        '''
        i is the numbering of optimal model wanted (int)
        target shoud be 'train', 'validation' or 'test'
        Return classification report, acc and auc of the stock/year wanted
        '''

        assert target in ['train', 'validation', 'test']
        

        report, acc = self.optimal.Separate_Evaluation(target, kwargs)

        return report, acc


    def Overall_Evaluation(self, target='test', verbose=True, threshold=0.6):

        '''
        i is the numbering of optimal model wanted (int)
        target shoud be 'train', 'validation' or 'test'
        threshold is the accuracy/auc score wanted
        Return the ratio of stocks performe worse than threshold 
        '''

        assert target in ['train', 'validation', 'test']
        assert threshold <= 1
        

        balance_acc, balance_auc = self.optimal.Overall_Evaluation(target, verbose=verbose, threshold=threshold)

        return balance_acc, balance_auc



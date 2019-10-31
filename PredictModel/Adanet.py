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
import pickle
import sys
import tensorflow as tf
import adanet


class Input_generator:

    def __init__(self, df_path, cluster_num):

        with open(f'{df_path}Cluster_{cluster_num}_classification_minmax0_Weekly', 'rb') as fp:
            self._df = pickle.load(fp)

        self._X_train = np.array(self._df[0]).astype(np.float32)
        self._Y_train = self._df[1]
        self._X_val = np.array(self._df[2]).astype(np.float32)
        self._Y_val = self._df[3]
        self._X_test = np.array(self._df[4]).astype(np.float32)
        self._Y_test = self._df[5]

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

        '''
        One Hot Encoding
        Not need to call independently
        '''

        onehot_encoder = OneHotEncoder(sparse=False)

        d = np.array(data).reshape(len(data), 1)
        Y = onehot_encoder.fit_transform(d)

        return Y
    
        
    def _get_dependent_variable(self):

        '''
        Transform Y to other form
        Not need to call independently
        '''
        
        try:
            self._check_status()
        except Exception:
            print('Invalid data shape. Check data size.')
            
            
        self._lab_train = np.array([item[2] for item in self.Y_train]).astype(np.int32)
        self._lab_val = np.array([item[2] for item in self.Y_val]).astype(np.int32)
        self._lab_test = np.array([item[2] for item in self.Y_test]).astype(np.int32)
        
        self._y_train = self._categorical_transform(self._lab_train)
        self._y_val = self._categorical_transform(self._lab_val)
        self._y_test = self._categorical_transform(self._lab_test)


    def transformation(self, batch_size, epochs):

        self._get_dependent_variable()

        train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                            x={"x": self._X_train},
                            y=self._lab_train.reshape(-1,1),
                            batch_size=batch_size,
                            num_epochs=epochs,
                            shuffle=False)

        adanet_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                            x={"x": self._X_train},
                            y=self._lab_train.reshape(-1,1),
                            batch_size=batch_size,
                            num_epochs=1,
                            shuffle=False)

        val_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                            x={"x": self._X_val},
                            y=self._lab_val.reshape(-1,1),
                            batch_size=batch_size,
                            num_epochs=1,
                            shuffle=False)

        test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                            x={"x": self._X_test},
                            y=self._lab_test.reshape(-1,1),
                            batch_size=batch_size,
                            num_epochs=1,
                            shuffle=False)

        return train_input_fn, adanet_input_fn, val_input_fn, test_input_fn 



class CNNBuilder(adanet.subnetwork.Builder):

    def __init__(self, conv_layer, dense_layer, conv_neuron, dense_neuron, dropout, learning_rate):

        
        self._conv_layer = conv_layer
        self._dense_layer = dense_layer
        self._conv_neuron = conv_neuron
        self._dense_neuron = dense_neuron
        self._drop_out = dropout
        self._learning_rate = learning_rate

    def build_subnetwork(self,
                        features,
                        logits_dimension,
                        training,
                        iteration_step,
                        summary,
                        previous_ensemble=None):
        
        x = list(features.values())[0]

        x = tf.layers.conv1d(x, filters=self._conv_neuron[0], 
                            kernel_size=3, 
                            activation=tf.nn.relu,  
                            padding='causal')
    
        for i in range(self._conv_layer - 1):
            x = tf.layers.conv1d(x, filters=self._conv_neuron[i+1], 
                                kernel_size=3, 
                                activation=tf.nn.relu,  
                                padding='causal')

            x = tf.layers.dropout(x, rate=self._drop_out[i])
            
        x = tf.layers.flatten(x)

        for i in range(self._dense_layer - 1):
            x = tf.layers.dense(x, units=self._dense_neuron[i], activation=tf.nn.relu)
            x = tf.layers.dropout(x, self._drop_out[i+self._conv_layer - 1])
        
        logits = tf.layers.dense(x, units=1, activation=None)


        complexity = tf.sqrt(tf.cast(self._conv_layer + self._dense_layer, dtype=tf.float32)) + tf.sqrt(tf.cast(sum(self._conv_neuron) + sum(self._dense_neuron), dtype=tf.float32))/10 - tf.sqrt(tf.cast(sum(self._drop_out), dtype=tf.float32))

        shared = {'conv_layer': self._conv_layer, 
                    'dense_layer': self._dense_layer, 
                    'conv_neuron': self._conv_neuron, 
                    'dense_neuron': self._dense_neuron, 
                    'dropout': self._drop_out,
                    'learning_rate': self._learning_rate}

        
        return adanet.Subnetwork(
            last_layer=x,
            logits=logits,
            complexity=complexity,
            shared=shared)

    
    def build_subnetwork_train_op(self,
                                  subnetwork,
                                  loss,
                                  var_list,
                                  labels,
                                  iteration_step,
                                  summary,
                                  previous_ensemble=None):


        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)

        return optimizer.minimize(loss=loss, var_list=var_list)


    def build_mixture_weights_train_op(self,
                                       loss,
                                       var_list,
                                       logits,
                                       labels,
                                       iteration_step, summary):

        return tf.no_op("mixture_weights_train_op")

    
    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        return f'CNN_{self._conv_layer}_{self._conv_neuron[0]}_{self._dense_layer}_{self._dense_neuron[0]}_{self._learning_rate}'
        


class CNNGenerator(adanet.subnetwork.Generator):

    def __init__(self):

        self._cnn_builder_fn = CNNBuilder


    def generate_candidates(self,
                                previous_ensemble,
                                iteration_number,
                                previous_ensemble_reports,
                                all_reports):

        conv_layer = 1
        dense_layer = 1
        dense_neuron = [64, 32]    
        dropout = [0.3, 0.3, 0.3, 0.3]
        conv_neuron = [32, 64, 64, 128]
        learning_rate = 0.001



        if previous_ensemble:
            conv_layer = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .shared['conv_layer'])

            dense_layer = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .shared['dense_layer'])

            dense_neuron = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .shared['dense_neuron'])

            conv_neuron = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .shared['conv_neuron'])

            dropout = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .shared['dropout'])

            learning_rate = tf.contrib.util.constant_value(
                previous_ensemble.weighted_subnetworks[-1]
                .subnetwork
                .shared['learning_rate'])

        if conv_layer == 4:
            new_conv_layer = conv_layer
            new_conv_neuron = [item+16 for item in conv_neuron]
            
        else:
            new_conv_layer = conv_layer + 1
            new_conv_neuron = [item+8 for item in conv_neuron]

        if dense_layer == 2:
            new_dense_layer = dense_layer
            new_dense_neuron = [item+16 for item in dense_neuron]
        else:
            new_dense_layer = dense_layer + 1   
            new_dense_neuron = [item+8 for item in dense_neuron]

        new_dropout = [item-0.05 for item in dropout]
        new_learning_rate = learning_rate + 0.001

        
        return [self._cnn_builder_fn(conv_layer=conv_layer, dense_layer=dense_layer, conv_neuron=conv_neuron, dense_neuron=dense_neuron, dropout=dropout, learning_rate=learning_rate), 
        self._cnn_builder_fn(conv_layer=new_conv_layer, dense_layer=new_dense_layer, conv_neuron=new_conv_neuron, dense_neuron=new_dense_neuron, dropout=new_dropout, learning_rate=new_learning_rate)]


class Train_adanet(Input_generator):

    def __init__(self, df_path, cluster_num, ada_steps, epoch, batch_size, CNNparam, config_name, seed):

        super(Train_adanet, self).__init__(df_path, cluster_num)

        self.max_step = self._X_train.shape[0] * epoch//(batch_size)
        self.iteration = self.max_step // ada_steps
        self.epoch = epoch
        self.batch_size = batch_size
        self._CNNparam = CNNparam
        self._config = config_name
        self._seed = seed

        self.train_input_fn, self.adanet_input_fn, self.val_input_fn, self.test_input_fn = self.transformation(self.batch_size, self.epoch)

    
    def make_config(self, experiment_name):
        # Estimator configuration.
        return tf.estimator.RunConfig(
            save_checkpoints_steps=5000,
            save_summary_steps=5000,
            tf_random_seed=self._seed,
            model_dir=os.path.join('D:\\庫存健診開發\\model\\', experiment_name))


    
    def train(self):

        head = tf.contrib.estimator.binary_classification_head(loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

        self._estimator = adanet.Estimator(head=head,
                                            subnetwork_generator=CNNGenerator(),
                                            max_iteration_steps=self.iteration,
                                            evaluator=adanet.Evaluator(
                                                input_fn=self.adanet_input_fn,
                                                steps=None),
                                            adanet_loss_decay=.99,
                                            config=self.make_config(self._config))

        self.results, _ = tf.estimator.train_and_evaluate(
                                            self._estimator,
                                            train_spec=tf.estimator.TrainSpec(
                                                input_fn=self.train_input_fn,
                                                max_steps=self.max_step),
                                            eval_spec=tf.estimator.EvalSpec(
                                                input_fn=self.val_input_fn,
                                                steps=None))

    
    def ensemble_architecture(self):

        architecture = self.results["architecture/adanet/ensembles"]
    # The architecture is a serialized Summary proto for TensorBoard.
        summary_proto = tf.summary.Summary.FromString(architecture)

        return summary_proto.value[0].tensor.string_val[0]
                        

    def predict(self, target='test'):

        assert target in ['train', 'validation', 'test']

        pred = []

        if target == 'train':
            predictions = self._estimator.predict(input_fn=self.adanet_input_fn)

        elif target == 'validation':
            predictions = self._estimator.predict(input_fn=self.val_input_fn)

        elif target == 'test':
            predictions = self._estimator.predict(input_fn=self.test_input_fn)

        for i, val in enumerate(predictions):
            pred.append(val['class_ids'][0])

        return np.array(pred)


    def Evaluation(self, target='test'):

        assert target in ['train', 'validation', 'test']

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



    def Separate_Evaluation(self, target='test', **kwargs):

        '''
        target shoud be 'train', 'validation' or 'test'
        kwargs should only contain 'year', 'stock' or 'prediction'
        Return classification report, acc and auc of the stock/year wanted
        '''
        
        assert target in ['train', 'validation', 'test']
        assert len(kwargs) <= 3


        if 'prediction' in kwargs:
            prediction = kwargs['prediction']

            if target == 'train':
                Real = self._Y_train
            elif target == 'validation':
                Real = self._Y_val
            else:
                Real = self._Y_test
        
        else:
            if target == 'train':
                prediction = self.predict(target)
                Real = self._Y_train
                
            elif target == 'validation':
                prediction = self.predict(target)
                Real = self._Y_val
                
            else:
                prediction = self.predict(target)
                Real = self._Y_test

    
        if (len(kwargs.keys()) == 1) or ((len(kwargs.keys()) == 2) and ('prediction' in kwargs)):
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

        y = [prediction[i] for i in y_index]
        

                
        target_names = ['down', 'up']

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y)
        auc = metrics.auc(fpr, tpr)
        
        report = classification_report(y_true, y, target_names=target_names)
        accuracy = metrics.accuracy_score(y_true, y)
        
        return report, accuracy, auc


    def Overall_Evaluation(self, target='test', threshold=0.6, verbose=True):

        '''
        target shoud be 'train', 'validation' or 'test'
        threshold is the accuracy/auc score wanted
        Return the ratio of stocks performe worse than threshold 
        '''

        assert target in ['train', 'validation', 'test']
        assert threshold <= 1


        if target  == 'train':
            stock = list(set([sublist[1] for sublist in self._Y_train]))

        elif target == 'validation':
            stock = list(set([sublist[1] for sublist in self._Y_val]))

        else:
            stock = list(set([sublist[1] for sublist in self._Y_test]))

        prediction = self.predict(target)

        acc_list = []
        auc_list = []

        if verbose:
            for i, s in enumerate(tqdm(stock)):
                r, acc, auc = self.Separate_Evaluation(target, stock=s, prediction=prediction)
                acc_list.append(acc)
                auc_list.append(auc)

        else:
            for i, s in enumerate(stock):
                r, acc, auc = self.Separate_Evaluation(target, stock=s, prediction=prediction)
                acc_list.append(acc)
                auc_list.append(auc)
            


        rate_acc = len([0 for i in acc_list if i < threshold])/len(acc_list)
        rate_auc = len([0 for i in auc_list if i < threshold])/len(auc_list)

        return rate_acc, rate_auc 

    
        

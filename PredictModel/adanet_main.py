import numpy as np 
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import Adanet

import pickle
import random
import time
import tensorflow as tf

def main():

    path = 'D:\\庫存健診開發\\data\\Training\\processed\\DL\\'
    cluster_num=0

    results = pd.DataFrame(columns = ['auc', 'acc', 'balance_acc', 'balance_auc'], index = ['train', 'val', 'test'])

    classifier = Adanet.Train_adanet(df_path=path, cluster_num=cluster_num, ada_steps=200, epoch=2000, batch_size=256, config_name='trial_1', penalty=0.003)

    classifier.CNN_train()

    auc_val, acc_val, report_val = classifier.Evaluation(target='validation')
    auc_test, acc_test, report_test = classifier.Evaluation(target='test')
    auc_train, acc_train, report_train = classifier.Evaluation(target='train')
    balance_acc_val, balance_auc_val = classifier.Overall_Evaluation(target='validation', verbose=False)
    balance_acc_test, balance_auc_test = classifier.Overall_Evaluation(target='test', verbose=False)
    balance_acc_train, balance_auc_train = classifier.Overall_Evaluation(target='train', verbose=False)

    results.loc['train', :] = auc_train, acc_train, balance_acc_train, balance_auc_train
    results.loc['val', :] = auc_val, acc_val, balance_acc_val, balance_auc_val
    results.loc['test', :] = auc_test, acc_test, balance_acc_test, balance_auc_test

    results.to_csv(f'D:\\庫存健診開發\\data\\adanet\\cluster_{cluster_num}.csv', index=True)


if __name__ == "__main__":
	main()

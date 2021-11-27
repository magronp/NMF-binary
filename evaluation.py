#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import load_tp_data_as_binary_csr, my_ndcg


# Define some parameters
curr_dataset = 'tp_med/'
params = {'data_dir': 'data/' + curr_dataset,
          'out_dir': 'outputs/' + curr_dataset,
          'batch_size': 1000,
          }

#  Get the number of songs and users in the training dataset (leave 5% of the songs for out-of-matrix prediction)
n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
n_songs = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

# Load the training and validation data
train_data = load_tp_data_as_binary_csr(params['data_dir'] + 'train.num.csv', shape=(n_users, n_songs))[0]
val_data = load_tp_data_as_binary_csr(params['data_dir'] + 'val.num.csv', shape=(n_users, n_songs))[0]
test_data = load_tp_data_as_binary_csr(params['data_dir'] + 'test.num.csv', shape=(n_users, n_songs))[0]

# Random predictions
pred_data = np.random.uniform(0, 1, (n_users, n_songs))
ndcg_mean = my_ndcg(test_data, pred_data, batch_users=params['batch_size'], k=50, leftout_ratings=train_data + val_data)[0]
print('random : ', ndcg_mean * 100)

# Load the trained model, compute predictions and score
for model_name in ['wmf', 'pf', 'bmf_em', 'bmf_mm']:
    factors = np.load(params['out_dir'] + model_name + '_model.npz')
    W, H, hypp = factors['W'], factors['H'], factors['hyper_params']
    pred_data = W.dot(H.T)
    ndcg_mean = my_ndcg(test_data, pred_data, batch_users=params['batch_size'], k=50, leftout_ratings=train_data + val_data)[0]
    print(model_name, ': ', ndcg_mean * 100, 'opt hyperparams:', hypp)

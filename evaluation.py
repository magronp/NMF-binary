#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from helpers.metrics import my_ndcg, my_ndcg_out
from helpers.algos import pure_content_recom
from helpers.data_feeder import load_tp_data_as_csr
import pandas as pd
from sklearn.preprocessing import scale
from scipy import sparse
import os
os.environ['OMP_NUM_THREADS'] = '10'

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def test_in_matrix(params, content='nocontent'):

    #  Get the number of songs and users in the 'in-matrix' dataset (leave 5% of the songs for out-of-matrix)
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_in = int(0.95 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Load the data
    train_data = load_tp_data_as_csr(os.path.join(params['data_dir'], 'in.train.num.csv'), shape=(n_users, n_songs_in))[0]
    vad_data = load_tp_data_as_csr(os.path.join(params['data_dir'], 'in.vad.num.csv'), shape=(n_users, n_songs_in))[0]
    test_data = load_tp_data_as_csr(os.path.join(params['data_dir'], 'in.test.num.csv'), shape=(n_users, n_songs_in))[0]

    # Load the WMF parameters and estimate the ratings
    params_wmf = np.load(params['out_dir'] + 'wmf_' + content + '.npz')
    W, H = params_wmf['W'], params_wmf['H']
    pred_ratings = W.dot(H.T)

    # Compute score
    ndcg_mean, ndcg_std = my_ndcg(test_data, pred_ratings, batch_users=params['batch_size'],
                                  leftout_ratings=train_data + vad_data)

    return ndcg_mean, ndcg_std


def test_out_matrix(params, content='avd'):

    #  Get the number of songs and users in the 'in-matrix' dataset (leave 5% of the songs for out-of-matrix)
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    n_songs_out = n_songs_total - int(0.95 * n_songs_total)

    # Load the data (out of matrix test set)
    test_data, rows_test, cols_test = load_tp_data_as_csr(os.path.join(params['data_dir'], 'out.test.num.csv'),
                                                          shape=(n_users, n_songs_total))
    cols_test -= cols_test.min()
    test_data = sparse.csr_matrix((test_data.data, (rows_test, cols_test)), dtype=np.int16, shape=(n_users, n_songs_out))

    # Predict the ratings
    if content == 'purecontent':
        pred_ratings = pure_content_recom(params)
    else:
        # Load the WMF parameters
        params_wmf = np.load(params['out_dir'] + 'wmf_' + content + '.npz')
        W, B = params_wmf['W'], params_wmf['B']

        # Load content features (and scale)
        Z = pd.read_csv(params['data_dir'] + 'out.' + content + '.num.csv').to_numpy()
        Z = Z[Z[:, 0].argsort()]
        Z = scale(np.delete(Z, 0, axis=1), axis=0)

        # Compute song attributes for the out-of-matrix songs, and then predict the ratings
        H = np.matmul(Z, B)
        pred_ratings = np.matmul(W, H.T)

    # Compute score
    ndcg_mean, ndcg_std = my_ndcg_out(test_data, pred_ratings, batch_users=params['batch_size'])

    return ndcg_mean, ndcg_std


def test_all_models(params):

    # Initialize metric dictionaries
    ndcg_in, ndcg_out = {}, {}

    # Loop over all WMF models for in-matrix recommendation
    for c in ['nocontent', 'avd', 'essentia']:
        ndcg_mean, ndcg_std = test_in_matrix(params, content=c)
        ndcg_in[c] = [ndcg_mean, ndcg_std]

    # Loop over methods for out-of-matrix recommendation
    for c in ['purecontent', 'avd', 'essentia']:
        ndcg_mean, ndcg_std = test_out_matrix(params, content=c)
        ndcg_out[c] = [ndcg_mean, ndcg_std]

    # Record the metrics
    json.dump(ndcg_in, open(params['out_dir'] + 'ndcg_in.json', 'w'))
    json.dump(ndcg_out, open(params['out_dir'] + 'ndcg_out.json', 'w'))

    return


def display_results(out_dir='outputs/'):

    # Load the data
    ndcg_in = json.load(open(out_dir + 'ndcg_in.json'))
    ndcg_out = json.load(open(out_dir + 'ndcg_out.json'))

    print("----- NDCG - In matrix -------")
    for c in ['nocontent', 'avd', 'essentia']:
        print(c, ndcg_in[c])

    print("----- NDCG - Ouf of matrix -------")
    for c in ['purecontent', 'avd', 'essentia']:
        print(c, ndcg_out[c])

    return


if __name__ == '__main__':

    # Set the parameters
    params = {'data_dir': 'data/',
              'out_dir': 'outputs/',
              'num_factors': 50,
              'num_iters': 20,
              'batch_size': 5000,
              }

    # Run the evaluation for all the models
    test_all_models(params)

    # Display the results
    display_results(params['out_dir'])

# EOF

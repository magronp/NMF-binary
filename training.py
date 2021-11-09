#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import itertools
from sklearn.preprocessing import scale
from helpers.data_feeder import get_confidence, load_tp_data_as_csr
from helpers.algos import factorize
from helpers.metrics import my_ndcg
import os
os.environ['OMP_NUM_THREADS'] = '1'  # to not conflict with joblib

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def train_val(params, content='nocontent'):

    #  Get the number of songs and users in the training dataset (leave 5% of the songs for out-of-matrix prediction)
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_total = len(open(params['data_dir'] + 'unique_sid.txt').readlines())
    n_songs_train = int(0.95 * n_songs_total)

    # Load the training data
    train_data = load_tp_data_as_csr('data/in.train.num.csv', shape=(n_users, n_songs_train))[0]

    # Get the confidence
    conf = get_confidence(train_data, alpha=2.0, epsilon=1e-6)

    # Validation data
    vad_data, rows_vad, cols_vad = load_tp_data_as_csr('data/in.vad.num.csv',  shape=(n_users, n_songs_train))

    # Load content if any, remove the first column and scale to 0 mean and unit variance)
    Z = None
    if not(content == 'nocontent'):
        Z = pd.read_csv('data/in.' + content + '.num.csv').to_numpy()
        Z = Z[Z[:, 0].argsort()]
        Z = scale(np.delete(Z, 0, axis=1), axis=0)

    opt_ndgc = 0

    for i in range(params['rand_search_size']):

        # Get the hyper-parameters
        lW, lH = params['list_lambda_WH'][i]
        print("lambda_W: %.3f | lambda_H: %.3f" % (lW, lH))
        start_time = time.time()

        # WMF
        print("Factorization on the training set...")
        W, H, B = factorize(conf, params['n_factors'], Z=Z, n_iters=params['n_iters'], lambda_W=lW, lambda_H=lH,
                            lambda_B=0.01, dtype='float32', batch_size=params['batch_size'], n_jobs=-1)

        # Cross validation with NDCG@k: faster than NDCG but yields similar results in terms of metric ordering
        print('NDCG on the validation set...')
        # Predict the ratings and compute the score
        pred_ratings = W.dot(H.T)
        aux = my_ndcg(vad_data, pred_ratings, batch_users=params['batch_size'], k=50, leftout_ratings=train_data)
        ndcg_mean = aux[0]
        tot_time = time.time() - start_time
        print("NDCG: %.5f ---- total time %.5f " % (ndcg_mean, tot_time))

        # Check if the performance is better: save parameters
        if ndcg_mean > opt_ndgc:
            np.savez(params['out_dir'] + 'wmf_' + content + '_big.npz', W=W, H=H, B=B, lambda_W=lW, lambda_H=lH)
            opt_ndgc = ndcg_mean


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Define the hyper-parameters on which perform randomized search
    list_lambda_WH = np.random.permutation(list(itertools.product(np.logspace(-3, 3, 7), np.logspace(-3, 3, 7))))

    # Define some parameters
    params = {'data_dir': 'data/',
              'out_dir': 'outputs/',
              'n_factors': 50,
              'n_iters': 20,
              'batch_size': 10000,
              'list_lambda_WH': list_lambda_WH,
              'rand_search_size': 1
              }

    # Train the WMF model with or without content
    for cont in ['nocontent', 'avd', 'essentia']:
        loadlambd = np.load(params['out_dir'] + 'wmf_' + cont + '.npz')
        params['list_lambda_WH'] = [[loadlambd['lambda_W'], loadlambd['lambda_W']]]
        train_val(params, content=cont)

# EOF

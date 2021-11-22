#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import itertools
from helpers.wmf import factorize_wmf
from helpers.bin_nmf import train_nmf_binary
from helpers.pf_vi import pf
from helpers.functions import create_folder, load_tp_data_as_binary_csr, my_ndcg
import os
os.environ['OMP_NUM_THREADS'] = '1'  # to not conflict with joblib


def get_factorization(train_data, params, model_name, hypp):

    # Allocate W and H to avoid warning
    W, H = None, None

    if model_name == 'wmf':
        W, H = factorize_wmf(train_data, params['n_factors'], n_iters=params['n_iters'], lambda_W=hypp[0],
                             lambda_H=hypp[1],
                             dtype='float32', batch_size=params['batch_size'], n_jobs=-1, init_std=0.01)
    elif model_name == 'bmf_mm' or model_name == 'bmf_em':
        W, H = train_nmf_binary(train_data, n_factors=params['n_factors'], n_iters=params['n_iters'],
                                prior_alpha=hypp[0], prior_beta=hypp[1])
    elif model_name == 'pf':
        model_pf = pf(K=params['n_factors'], alphaW=hypp, alphaH=hypp)
        model_pf.fit(train_data, opt_hyper=['beta'], precision=0, max_iter=params['n_iters'], save=False)
        W, H = model_pf.Ew, model_pf.Eh

    return W, H


def training_validation(params, list_hyperparams, model_name='wmf'):

    # Create the output folder if needed
    create_folder(params['out_dir'])

    #  Get the number of songs and users in the training dataset (leave 5% of the songs for out-of-matrix prediction)
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs = len(open(params['data_dir'] + 'unique_sid.txt').readlines())

    # Load the training and validation data
    train_data = load_tp_data_as_binary_csr(params['data_dir'] + 'train.num.csv', shape=(n_users, n_songs))[0]
    val_data = load_tp_data_as_binary_csr(params['data_dir'] + 'val.num.csv', shape=(n_users, n_songs))[0]

    # initialize the optimal ndcg for validation
    val_ndcg = []
    opt_ndgc = 0
    n_hypp = len(list_hyperparams)

    for ih, hypp in enumerate(list_hyperparams):

        # Factorization
        print('----------- Hyper parameters ', ih+1, ' / ', n_hypp)
        print("Factorization on the training set...")
        W, H = get_factorization(train_data, params, model_name, hypp)

        # Validation with NDCG@50
        pred_data = W.dot(H.T)
        ndcg_mean = my_ndcg(val_data, pred_data, batch_users=params['batch_size'], k=50, leftout_ratings=train_data)[0]
        print('\n NDCG on the validation set: %.2f' % (ndcg_mean * 100))
        val_ndcg.append([hypp, ndcg_mean])

        # Check if the performance is better: save the model and record the corresponding hyper parameters
        if ndcg_mean > opt_ndgc:
            np.savez(params['out_dir'] + model_name + '_model.npz', W=W, H=H, hyper_params=hypp)
            opt_ndgc = ndcg_mean

    # Store the validation NDCG over hyperparameters
    np.savez(params['out_dir'] + model_name + '_val.npz', val_ndcg=val_ndcg)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Define the common parameters
    curr_dataset = 'tp_big/'

    params = {'data_dir': 'data/' + curr_dataset,
              'out_dir': 'outputs/' + curr_dataset,
              'n_factors': 50,
              'n_iters': 20,
              'batch_size': 1000,
              }

    # WMF
    rand_search_size = 10
    list_hyperparams = np.random.permutation(list(itertools.product(np.logspace(-3, 3, 7), np.logspace(-3, 3, 7))))
    list_hyperparams = list_hyperparams[:rand_search_size]
    training_validation(params, list_hyperparams, model_name='wmf')

    # PF
    list_hyperparams = [.001, .01, .1, 1, 10, 100, 1000]
    training_validation(params, list_hyperparams, model_name='pf')

    # Binary NMF - MM
    list_alpha = [0.01, 0.1, 1, 10]
    list_beta = list_alpha
    list_hyperparams = list(itertools.product(list_alpha, list_beta))
    training_validation(params, list_hyperparams, model_name='bmf_mm')

    # Binary NMF - EM (no prior)
    list_alpha = [1]
    list_beta = list_alpha
    list_hyperparams = list(itertools.product(list_alpha, list_beta))
    training_validation(params, list_hyperparams, model_name='bmf_em')


"""
# Plot the validation NDCG for the binary NMF model
ndcg_val = np.load(params['out_dir'] + 'bmf_mm_ndcg_val.npz')['ndcg_val']
plt.figure()
plt.imshow(ndcg_val)
plt.show()
"""

# EOF

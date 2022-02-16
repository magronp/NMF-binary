#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import pyreadr
from helpers.functions import create_folder, get_perplexity, build_split_masks
from helpers.nbmf import train_nbmf
from helpers.lpca import train_lpca


def model_fitting(data, train_mask, n_factors, prior_alpha, prior_beta, model='NBMF', max_iter=20, eps=1e-8):

    if 'NBMF' in model:
        Y = data.to_numpy()
        W, H, loss, tot_time = train_nbmf(Y, mask=train_mask, n_factors=n_factors, max_iter=max_iter,
                                          prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps)
        # Get predictions
        Y_hat = np.dot(W, H.T)
    else:
        W, H, Y_hat, tot_time = train_lpca(data, n_factors, max_iter, mask_leftout=1 - train_mask)
        loss = None

    return W, H, Y_hat, tot_time, loss


def traininig_with_validation(data, train_mask, val_mask, list_nfactors, list_alpha, list_beta,
                              dataset_output_dir, model='NBMF', max_iter=20, eps=1e-8):

    # initialize the optimal perplexity for validation
    opt_pplx = float(np.inf)
    
    # Initialize the array for storing validation perplexity
    nk, n_alpha, n_beta = len(list_nfactors), len(list_alpha), len(list_beta)
    val_pplx = np.zeros((nk, n_alpha, n_beta))
    
    # Counter
    n_hypp = n_alpha * n_beta * nk
    ih = 1

    for ik, n_factors in enumerate(list_nfactors):
        for ia, prior_alpha in enumerate(list_alpha):
            for ib, prior_beta in enumerate(list_beta):
                
                print('----------- Hyper parameters ', ih, ' / ', n_hypp)
                
                # Model fitting
                W, H, Y_hat, tot_time, loss = model_fitting(data, train_mask, n_factors, prior_alpha, prior_beta,
                                                            model=model, max_iter=max_iter, eps=eps)

                # Perplexity on the validation set
                perplx = get_perplexity(data.to_numpy(), Y_hat, mask=val_mask)
                print('\n Val perplexity: %.2f' % perplx)
                val_pplx[ik, ia, ib] = perplx
        
                # Check if the performance is better: save the model and record the corresponding hyper parameters
                if perplx < opt_pplx:
                    np.savez(dataset_output_dir + model + '_model.npz', W=W, H=H, Y_hat=Y_hat,
                             hyper_params=(n_factors, prior_alpha, prior_beta), time=tot_time, loss=loss)
                    opt_pplx = perplx
                
                # Update counter
                ih += 1

    # Store the validation perplexity for all hyperparameters
    np.savez(dataset_output_dir + model + '_val.npz', val_pplx=val_pplx, list_hyper=(list_nfactors, list_alpha, list_beta))

    return


def train_test_init(data, train_mask, test_mask, hyper_params, dataset_output_dir, model='NBMF', max_iter=20, eps=1e-8, n_init=10):

    # Initialize the array for storing perplexity and comp time over initializations
    test_pplx = np.zeros((n_init))
    test_time = np.zeros((n_init))

    # Loop over random initializations
    for ind_i in range(n_init):

        # Model fitting
        _, _, Y_hat, tot_time, _ = model_fitting(data, train_mask, int(hyper_params[0]), hyper_params[1], hyper_params[2],
                                                    model=model, max_iter=max_iter, eps=eps)

        # Get the perplexity on the test set and store it
        perplx = get_perplexity(data.to_numpy(), Y_hat, mask=test_mask)
        test_pplx[ind_i] = perplx
        test_time[ind_i] = tot_time

    # Save the perplexity and computation time
    np.savez(dataset_output_dir + model + '_test_init.npz', test_pplx=test_pplx, test_time=test_time)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Define general paths, all datasets and models
    data_dir = 'data/'
    out_dir = 'outputs/'
    models = ['NBMF-EM', 'NBMF-MM', 'logPCA']
    datasets = ['animals', 'paleo', 'lastfm']
    #datasets = ['animals']
    prop_train, prop_val = 0.7, 0.15

    # Hyperparameters
    max_iter = 2000
    eps = 1e-8
    n_init = 10
    list_nfactors = [2, 4, 8, 16]

    # Loop over datasets
    for dataset in datasets:

        # Load the data and masks
        data = pyreadr.read_r(data_dir + dataset + '.rda')[dataset]
        train_mask, val_mask, test_mask = build_split_masks(data.shape, prop_train=prop_train, prop_val=prop_val)
        np.savez(data_dir + dataset + '_split.npz', train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        # Define and create the output directory
        dataset_output_dir = out_dir + dataset + '/'
        create_folder(dataset_output_dir)

        # Training with validation - NMBF-EM (no prior)
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, [1.], [1.], dataset_output_dir,
                                  model='NBMF-EM', max_iter=max_iter, eps=eps)

        # Training with validation - NMBF-MM
        list_alpha = np.linspace(1, 5, 9)
        list_beta = list_alpha
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, list_alpha, list_beta, dataset_output_dir,
                                  model='NBMF-MM', max_iter=max_iter, eps=eps)

        """ 
        # Same but with alt priors
        list_alpha = np.linspace(0, 2, 11)
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, list_alpha, [None], dataset_output_dir,
                                  model='nbmf_alt', max_iter=max_iter, eps=eps)
        """

        # Train the logistic PCA model
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, [None], [None], dataset_output_dir,
                                  model='logPCA', max_iter=max_iter, eps=eps)

        # After validation, compute perplexity on the test set with many random initializations
        for model in models:

            hyper_params = np.load(dataset_output_dir + model + '_model.npz', allow_pickle=True)['hyper_params']

            # Training and test with several random initialization
            train_test_init(data, train_mask, test_mask, hyper_params, dataset_output_dir, model=model, max_iter=max_iter, eps=eps, n_init=n_init)

# EOF

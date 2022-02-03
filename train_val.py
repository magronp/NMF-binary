#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import create_folder, get_perplexity, build_split_masks
from helpers.nbmf import train_nbmf
from helpers.lpca import train_lpca
import pyreadr


def traininig_with_validation(data, train_mask, val_mask, list_nfactors, list_alpha, list_beta,
                              dataset_output_dir, model_name='bmf', max_iter=20, eps=1e-8):

    Y = data.to_numpy()

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
                if 'nbmf' in model_name:
                    W, H, _ = train_nbmf(Y, mask=train_mask, n_factors=n_factors, max_iter=max_iter,
                                      prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps)
                    # Get predictions
                    Y_hat = np.dot(W, H.T)
                else:
                    W, H, Y_hat = train_lpca(data, n_factors, max_iter, mask_leftout=1-train_mask)

                perplx = get_perplexity(Y, Y_hat, mask=val_mask)
                print('\n Val perplexity: %.2f' % (perplx))
                val_pplx[ik, ia, ib] = perplx
        
                # Check if the performance is better: save the model and record the corresponding hyper parameters
                if perplx < opt_pplx:
                    np.savez(dataset_output_dir + model_name + '_model.npz', W=W, H=H, Y_hat=Y_hat,
                             hyper_params=(n_factors, prior_alpha, prior_beta))
                    opt_pplx = perplx
                
                # Update counter
                ih += 1

    # Store the validation perplexity for all hyperparameters
    np.savez(dataset_output_dir + model_name + '_model_val.npz', val_pplx=val_pplx, list_hyper=(list_nfactors, list_alpha, list_beta))

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # General path
    data_dir = 'data/'
    out_dir = 'outputs/'

    # Hyperparameters
    max_iter = 2000
    eps = 1e-8
    list_nfactors = [2, 4, 8, 16]
    list_alpha = np.linspace(1, 20, 20)
    #list_alpha = np.linspace(1, 3, 11)
    list_beta = list_alpha

    # All datasets
    datasets = ['animals', 'paleo', 'lastfm']

    # Loop over datasets
    for my_dataset in datasets:

        # Load the data
        dataset_path = data_dir + my_dataset
        data = pyreadr.read_r(dataset_path + '.rda')[my_dataset]

        # Define and create the output directory
        dataset_output_dir = out_dir + my_dataset + '/'
        create_folder(dataset_output_dir)

        # Create train / val / test split in the form of binary masks (and record the test mask for evaluation)
        train_mask, val_mask, test_mask = build_split_masks(data.shape)
        np.savez(dataset_path + '_split.npz', train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        # Training with validation
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, list_alpha, list_beta, dataset_output_dir,
                                  model_name='nbmf', max_iter=max_iter, eps=eps)

        # Same but with no priors
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, [1.], [1.], dataset_output_dir,
                                  model_name='nbmf_noprior', max_iter=max_iter, eps=eps)

        # Train the logistic PCA model
        traininig_with_validation(data, train_mask, val_mask, list_nfactors, [1.], [1.], dataset_output_dir,
                                  model_name='lpca', max_iter=max_iter, eps=eps)

# EOF

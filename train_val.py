#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

from tqdm import tqdm
import numpy as np
from helpers.functions import create_folder, get_perplexity, build_split_masks
from bmf import train_nmf_binary_dense
import pyreadr
import itertools


def train_val_nmf_binary_dense(Y, train_mask, val_mask, list_hyperparams, dataset_output_dir,
                               model_name='bmf', n_iters=20, eps=1e-8):

    # initialize the optimal perplexity for validation
    val_pplx = []
    opt_pplx = float(np.inf)
    n_hypp = len(list_hyperparams)

    for ih, hypp in enumerate(list_hyperparams):

        prior_alpha, prior_beta, n_factors = hypp[0], hypp[1], hypp[2]

        # Factorization
        print('----------- Hyper parameters ', ih + 1, ' / ', n_hypp)
        print("Factorization on the training set...")
        W, H = train_nmf_binary_dense(Y, mask=train_mask, n_factors=n_factors, n_iters=n_iters,
                                      prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps)

        # Validation
        Y_hat = np.dot(W, H.T)
        perplx = get_perplexity(Y, Y_hat, mask=val_mask)
        print('\n Val perplexity: %.2f' % (perplx))
        val_pplx.append([hypp, perplx])

        # Check if the performance is better: save the model and record the corresponding hyper parameters
        if perplx < opt_pplx:
            np.savez(dataset_output_dir + model_name + '_model.npz', W=W, H=H, hyper_params=hypp)
            opt_pplx = perplx

    # Store the validation perplexity for all hyperparameters
    np.savez(dataset_output_dir + model_name + '_model_val.npz', val_pplx=val_pplx)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # General path
    data_dir = 'data/'
    out_dir = 'outputs/'

    # Load the data
    my_dataset = 'lastfm'
    dataset_path = data_dir + my_dataset
    Y = pyreadr.read_r(dataset_path + '.rda')[my_dataset].to_numpy()

    # Define and create the output directory
    dataset_output_dir = out_dir + my_dataset + '/'
    create_folder(dataset_output_dir)

    # Create train / val / test split in the form of binary masks (and record the test mask for evaluation)
    train_mask, val_mask, test_mask = build_split_masks(Y.shape)
    np.savez(dataset_path + '_split.npz', train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # Hyperparameters
    n_iters = 100
    eps = 1e-8
    list_nfactors = [8, 16, 32, 64, 128]
    list_alpha = np.linspace(1, 3, 11)
    list_beta = list_alpha
    list_hyperparams = list(itertools.product(list_alpha, list_beta, list_nfactors))

    # Training with validation
    train_val_nmf_binary_dense(Y, train_mask, val_mask, list_hyperparams, dataset_output_dir,
                               model_name='bmf', n_iters=n_iters, eps=eps)

    # Same but with no priors
    list_alpha = [1.]
    list_beta = list_alpha
    list_hyperparams = list(itertools.product(list_alpha, list_beta, list_nfactors))
    train_val_nmf_binary_dense(Y, train_mask, val_mask, list_hyperparams, dataset_output_dir,
                               model_name='bmf_noprior', n_iters=n_iters, eps=eps)

# EOF

#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

from tqdm import tqdm
import numpy as np
from helpers.functions import get_perplexity, nbmf_loss
import pyreadr
import time


def train_nbmf(Y, mask=None, n_factors=8, max_iter=10, prior_alpha=1., prior_beta=1., eps=1e-8):

    # Get the shapes
    n_users, n_songs = Y.shape
    
    # Initialize losses
    loss = []
    loss_prev = np.inf
    
    if mask is None:
        mask = np.ones_like(Y)

    start_time = time.time()

    # Mask the data to retain only what's in the training set (and precompute Y transpoed and 1-Yt)
    YT = Y.T * mask.T
    OneminusYT = (1 - Y.T) * mask.T
    Y = Y * mask

    # Initialize NMF matrices (respecting constraints)
    H = np.random.uniform(0, 1, (n_factors, n_songs))
    W = np.random.uniform(0, 1, (n_factors, n_users))
    Wsum = np.sum(W, axis=0)
    Wsum = np.repeat(Wsum[:, np.newaxis].T, n_factors, axis=0)
    W = W / Wsum

    # Beta prior
    if prior_beta is not None:
        A = np.ones_like(H) * (prior_alpha - 1)
        B = np.ones_like(H) * (prior_beta - 1)
    else:
        mu = np.sum(Y, axis=0) / Y.shape[0]
        pr_alpha = prior_alpha * mu
        pr_beta = prior_alpha * (1-mu)
        A = np.repeat(pr_alpha[np.newaxis, :], n_factors, axis=0)
        B = np.repeat(pr_beta[np.newaxis, :], n_factors, axis=0)
        
    for _ in tqdm(range(max_iter)):

        # Update on H
        WH = np.dot(W.T, H)
        numerator = H * np.dot(W, Y / (WH + eps)) + A
        denom2 = (1 - H) * np.dot(W, (1 - Y) / (1 - WH + eps)) + B
        H = numerator / (numerator + denom2)

        # Update on W
        WtHT = np.dot(H.T, W)
        W = W * (np.dot(H, YT / (WtHT + eps)) + np.dot(1 - H, OneminusYT / (1 - WtHT + eps))) / n_songs
        
        # Get the loss and convergence criterion
        # loss_new = nbmf_loss(Y, W, H, prior_alpha=prior_alpha, prior_beta=prior_beta, mask=mask, eps=eps)
        loss_new = nbmf_loss(Y, W, H, A, B, mask=mask, eps=eps)
        loss.append(loss_new)

        # Check if convergence has been reached
        if (loss_prev - loss_new) < 1e-5:
            break
        loss_prev = loss_new

    # Get the computation time
    tot_time = time.time() - start_time

    return W.T, H.T, loss, tot_time


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # General path
    data_dir = '../data/'

    # Load the data
    my_dataset = 'animals'
    dataset_path = data_dir + my_dataset + '.rda'
    Y = pyreadr.read_r(dataset_path)[my_dataset].to_numpy()

    # training on one set of hyperparameters (no masking, thus no train/val/test split)
    prior_alpha, prior_beta = 2, 2
    n_factors = 4
    max_iter = 2000
    eps = 1e-8
    W, H, loss, tot_time = train_nbmf(Y, n_factors=n_factors, max_iter=max_iter,
                            prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps)
    Y_hat = np.dot(W, H.T)
    perplx = get_perplexity(Y, Y_hat)
    print('\n Training perplexity:', perplx)

# EOF

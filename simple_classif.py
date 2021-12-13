#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import os
import time
from scipy import sparse
from tqdm import tqdm
from helpers.functions import load_tp_data_as_binary_csr, plot_hist_predictions, my_ndcg, plot_hist_predictions_list
import numpy as np
import json
import os
import pandas as pd
from helpers.functions import create_folder
import pyreadr


def train_nmf_binary_dense(Y, mask=None, n_factors=20, n_iters=20, prior_alpha=1., prior_beta=1., eps=1e-8):

    # Get the shapes
    n_users, n_songs = Y.shape

    if mask is None:
        mask = np.ones_like(Y)

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
    A = np.ones_like(H) * (prior_alpha - 1)
    B = np.ones_like(H) * (prior_beta - 1)

    for _ in tqdm(range(n_iters)):
        # Update on H
        WH = np.dot(W.T, H)
        numerator = H * np.dot(W, Y / (WH + eps)) + A
        denom2 = (1 - H) * np.dot(W, (1 - Y) / (1 - WH + eps)) + B
        H = numerator / (numerator + denom2)

        # Update on W
        WtHT = np.dot(H.T, W)
        W = W * (np.dot(H, YT / (WtHT + eps)) + np.dot(1 - H, OneminusYT / (1 - WtHT + eps))) / n_songs

    return W.T, H.T


def get_perplexity(Y, Y_hat, mask=None, eps=1e-8):

    if mask is None:
        mask = np.ones_like(Y)

    perplx = Y * np.log(Y_hat + eps) + (1-Y) * np.log(1 - Y_hat + eps)
    perplx = - np.sum(mask * perplx)

    return perplx


# Set random seed for reproducibility
np.random.seed(12345)

# Load the data
my_dataset = 'lastfm'
data_dir = 'data/'
data_dir_rda = data_dir + my_dataset + '/'
create_folder(data_dir_rda)
Y = pyreadr.read_r(data_dir + my_dataset + '.rda')[my_dataset].to_numpy()

# Create train / val / test split in the form of binary masks


# Define the parameters
curr_dataset = 'tp_small/'
data_dir = 'data/' + curr_dataset
prior_alpha, prior_beta = 1.1, 1.2
n_factors = 64
n_iters = 100
eps = 1e-8
comp_loss = True

# train
W, H = train_nmf_binary_dense(Y, mask=train_mask, n_factors=n_factors, n_iters=n_iters, prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps)

# test
Y_hat = np.dot(W, H.T)
perplx = get_perplexity(Y, Y_hat, test_mask)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import os
from scipy import sparse
from tqdm import tqdm
os.environ['OMP_NUM_THREADS'] = '1'  # to not conflict with joblib


def train_nmf_binary(Y, n_factors=20, n_iters=20, prior_alpha=1, prior_beta=1, eps=1e-8):
    """
    Meant for processing sparse matrices for Y
    """

    # Get the shapes
    n_users, n_songs = Y.shape

    # Precompute variants of the training data
    YT = Y.T
    One_minus_Y = sparse.csr_matrix(1-Y.toarray())
    One_minus_YT = One_minus_Y.T

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
        numerator = np.multiply(H, np.dot(W, Y / (WH + eps))) + A
        denom2 = np.multiply((1 - H), np.dot(W, One_minus_Y / (1 - WH + eps))) + B
        H = numerator / (numerator + denom2)

        # Update on W
        WtHT = np.dot(H.T, W)
        W = np.multiply(W, np.dot(H, YT / (WtHT + eps)) + np.dot(1 - H, One_minus_YT / (1 - WtHT + eps))) / n_songs

    return W.T, H.T


def train_nmf_binary_dense(Y, n_factors=20, n_iters=20, prior_alpha=1, prior_beta=1, eps=1e-8, verbose=True):

    # Get the shapes
    n_users, n_songs = Y.shape

    # Precompute the transposed training data
    YT = Y.T

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
        W = W * (np.dot(H, YT / (WtHT + eps)) + np.dot(1 - H, (1 - YT) / (1 - WtHT + eps))) / n_songs

    return W.T, H.T


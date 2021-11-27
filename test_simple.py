#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import os
from scipy import sparse
from helpers.functions import load_tp_data_as_binary_csr, plot_hist_predictions
from helpers.bin_nmf import train_nmf_binary_fast
from matplotlib import pyplot as plt

os.environ['OMP_NUM_THREADS'] = '1'  # to not conflict with joblib


# Set random seed for reproducibility
np.random.seed(12345)

# generate data
'''
n_users = 5
n_songs = 3
n_factors = 3
Hs = np.random.uniform(0, 1, (n_songs, n_factors))
Ws = np.random.uniform(0, 1, (n_users, n_factors))
Wsum = np.sum(Ws, axis=1)
Wsum = np.repeat(Wsum[:, np.newaxis], n_factors, axis=1)
Ws = Ws / Wsum
Y = np.dot(Ws, Hs.T)

'''
Ydense = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
Ydense = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
n_factors = 4
Y = sparse.csr_matrix(Ydense)

# Define the parameters
prior_alpha, prior_beta  = 1, 1
n_iters = 50
eps = 1e-8
W, H, loss, time_tot = train_nmf_binary_fast(Y, None, n_factors=n_factors, n_iters=n_iters, prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps, comp_loss=True)
Yhat = np.dot(W, H.T)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(Ydense, aspect='auto')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(Yhat, aspect='auto')
plt.colorbar()
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(W, aspect='auto')
plt.subplot(1, 2, 2)
plt.imshow(H.T, aspect='auto')
plt.show()
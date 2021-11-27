#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import os
import time
from scipy import sparse
from tqdm import tqdm
from helpers.functions import load_tp_data_as_binary_csr, plot_hist_predictions, my_ndcg

os.environ['OMP_NUM_THREADS'] = '1'  # to not conflict with joblib


def get_confidence(playcounts, alpha=2.0, epsilon=1e-6):
    conf = playcounts.copy()
    conf.data = 1 + alpha * np.log(1 + conf.data / epsilon)
    return conf


def train_nmf_binary_fast(Y, left_out_data=None, n_factors=20, n_iters=20, prior_alpha=1, prior_beta=1, eps=1e-8, comp_loss=False, val_data=None):
    """
    Meant for processing sparse matrices for Y
    """
    # Get the shapes
    n_users, n_songs = Y.shape

    # Acount for the left out data (val / test) needed for 1-Y
    One_minus_Y = sparse.csr_matrix(1-Y.toarray())
    if left_out_data is None:
        One_minus_Y_wo_val = One_minus_Y
    else:
        rows_omY, cols_omY = One_minus_Y.nonzero()
        rows_lo, cols_lo = left_out_data.nonzero()
        indx_omY = [(rows_omY[i], cols_omY[i]) for i in range(len(cols_omY))]
        indx_lo = [(rows_lo[i], cols_lo[i]) for i in range(len(cols_lo))]
        indx_comp = [x for x in indx_omY if not (x in indx_lo)]
        rows_comp = [indx_comp[i][0] for i in range(len(indx_comp))]
        cols_comp = [indx_comp[i][1] for i in range(len(indx_comp))]
        One_minus_Y_wo_val = sparse.csr_matrix((np.ones(len(indx_comp)),
                                                (np.array(rows_comp, dtype=np.int32), np.array(cols_comp, dtype=np.int32))),
                                               dtype=np.int16, shape=One_minus_Y.shape)

    # Precompute the transposed version of the data
    YT = Y.T
    One_minus_YT_wo_val = One_minus_Y_wo_val.T

    # Initialize NMF matrices (respecting constraints)
    H = np.random.uniform(0, 1, (n_songs, n_factors))
    W = np.random.uniform(0, 1, (n_users, n_factors))
    Wsum = np.sum(W, axis=1)
    Wsum = np.repeat(Wsum[:, np.newaxis], n_factors, axis=1)
    W = W / Wsum

    # Beta prior
    A = np.ones_like(H) * (prior_alpha - 1)
    B = np.ones_like(H) * (prior_beta - 1)

    loss, ndcg_val = [], []
    start_time = time.time()
    
    for _ in tqdm(range(n_iters)):

        # Update on H
        WH = np.dot(W, H.T)

        if not(val_data is None):
            ndcg_mean = my_ndcg(val_data, WH, k=50, leftout_ratings=Y)[0]
            ndcg_val.append(ndcg_mean)
            
        # Compute the loss
        if comp_loss:
            loss.append(-np.sum(Y.multiply(np.log(WH+eps)) + One_minus_Y_wo_val.multiply(np.log(1-WH+eps))) -
                        np.sum(A * np.log(H+eps) + B * np.log(1-H+eps)))

        numerator = H * Y.multiply(1 / (WH + eps)).T.dot(W) + A
        denom2 = (1 - H) * One_minus_Y_wo_val.multiply(1 / (1 - WH + eps)).T.dot(W) + B
        H = numerator / (numerator + denom2)
        
        # Update on W
        WtHT = np.dot(H, W.T)
        W = W * (YT.multiply(1 / (WtHT + eps)).T.dot(H) + One_minus_YT_wo_val.multiply(1 / (1 - WtHT + eps)).T.dot(1 - H)) / n_songs
        
    time_tot = time.time() - start_time
    
    W, H = np.array(W), np.array(H)
    
    return W, H, loss, time_tot, ndcg_val


def train_nmf_binary(Y, left_out_data, n_factors=20, n_iters=20, prior_alpha=1, prior_beta=1, eps=1e-8, comp_loss=False):
    """
    Meant for processing sparse matrices for Y
    """
    # Get the shapes
    n_users, n_songs = Y.shape

    # Acount for the left out data (val / test) needed for 1-Y
    One_minus_Y = sparse.csr_matrix(1-Y.toarray())
    rows_omY, cols_omY = One_minus_Y.nonzero()
    rows_lo, cols_lo = left_out_data.nonzero()
    indx_omY = [(rows_omY[i], cols_omY[i]) for i in range(len(cols_omY))]
    indx_lo = [(rows_lo[i], cols_lo[i]) for i in range(len(cols_lo))]
    indx_comp = [x for x in indx_omY if not (x in indx_lo)]
    rows_comp = [indx_comp[i][0] for i in range(len(indx_comp))]
    cols_comp = [indx_comp[i][1] for i in range(len(indx_comp))]
    One_minus_Y_wo_val = sparse.csr_matrix((np.ones(len(indx_comp)),
                                            (np.array(rows_comp, dtype=np.int32), np.array(cols_comp, dtype=np.int32))),
                                           dtype=np.int16, shape=One_minus_Y.shape)

    # Precompute the transposed version of the data
    YT = Y.T
    One_minus_YT_wo_val = One_minus_Y_wo_val.T

    # Initialize NMF matrices (respecting constraints)
    H = np.random.uniform(0, 1, (n_factors, n_songs))
    W = np.random.uniform(0, 1, (n_factors, n_users))
    Wsum = np.sum(W, axis=0)
    Wsum = np.repeat(Wsum[:, np.newaxis].T, n_factors, axis=0)
    W = W / Wsum

    # Beta prior
    A = np.ones_like(H) * (prior_alpha - 1)
    B = np.ones_like(H) * (prior_beta - 1)

    loss = []
    start_time = time.time()

    for _ in tqdm(range(n_iters)):

        # Update on H
        WH = np.dot(W.T, H)

        # Compute the loss
        if comp_loss:
            loss.append(-np.sum(Y.multiply(np.log(WH)) + One_minus_Y_wo_val.multiply(np.log(1-WH))) -
                        np.sum(np.multiply(A, np.log(H)) + np.multiply(B, np.log(1-H))))

        numerator = np.multiply(H, np.dot(W, Y / (WH + eps))) + A
        denom2 = np.multiply((1 - H), np.dot(W, One_minus_Y_wo_val / (1 - WH + eps))) + B
        H = numerator / (numerator + denom2)

        # Update on W
        WtHT = np.dot(H.T, W)
        W = np.multiply(W, np.dot(H, YT / (WtHT + eps)) + np.dot(1 - H, One_minus_YT_wo_val / (1 - WtHT + eps))) / n_songs

    time_tot = time.time() - start_time
    
    W, H = np.array(W), np.array(H)
    
    return W.T, H.T, loss, time_tot


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


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Define the parameters
    curr_dataset = 'tp_small/'
    data_dir = 'data/' + curr_dataset
    prior_alpha, prior_beta  = 1, 1
    n_factors = 64
    n_iters = 10
    eps = 1e-8
    comp_loss = True
    
    #  Get the number of songs and users in the training dataset (leave 5% of the songs for out-of-matrix prediction)
    n_users = len(open(data_dir + 'unique_uid.txt').readlines())
    n_songs = len(open(data_dir + 'unique_sid.txt').readlines())

    # Load the training and validation data
    train_data = load_tp_data_as_binary_csr(data_dir + 'train.num.csv', shape=(n_users, n_songs))[0]
    val_data = load_tp_data_as_binary_csr(data_dir + 'val.num.csv', shape=(n_users, n_songs))[0]
    test_data = load_tp_data_as_binary_csr(data_dir + 'test.num.csv', shape=(n_users, n_songs))[0]
    left_out_data = val_data + test_data
    Y = train_data
    
    W, H, loss, time_tot, ndcg_val = train_nmf_binary_fast(Y, left_out_data, n_factors=n_factors, n_iters=n_iters, prior_alpha=prior_alpha, prior_beta=prior_beta, eps=eps, comp_loss=comp_loss)
    #plt.semilogy(loss)
    # Vizualization
    #plot_hist_predictions(W, H, val_data, len_max=200)
    
    # checker viz avec prior, ptetr ça change
# EOF

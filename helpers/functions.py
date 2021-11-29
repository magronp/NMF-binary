#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import bottleneck as bn
import numpy as np
from scipy import sparse
import pandas as pd
import os
from matplotlib import pyplot as plt


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def load_tp_data_as_binary_csr(csv_file, shape):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    sparse_tp = sparse.csr_matrix((count, (rows, cols)), dtype=np.int16, shape=shape)
    # Binarize the data
    sparse_tp.data = np.ones_like(sparse_tp.data)
    return sparse_tp, rows, cols


def pred_data_from_WH(W, H, true_data, len_max=1000):
    
    # get rows and cols indices for the data and shuffle them
    rows, cols = true_data.nonzero()
    my_perm = np.random.permutation(len(rows))
    rows, cols = rows[my_perm], cols[my_perm]
    # initialize the array for storing the prediction
    pred_len = np.minimum(len(rows), len_max)
    pred_data = np.zeros(pred_len)
    # calculate predictions as WH
    for i in range(pred_len):
       pred_data[i] = np.sum(W[rows[i], :] * H[cols[i], :])
    
    return pred_data


def plot_hist_predictions(W, H, Y, len_max=1000):
    
    pred_data1 = pred_data_from_WH(W, H, Y, len_max)
    pred_data0 = pred_data_from_WH(W, H, sparse.csr_matrix(1-Y.toarray()), len_max)
    plt.figure()
    plt.hist(pred_data1, bins=50)
    plt.hist(pred_data0, bins=50)
    plt.legend(['ones', 'zeros'])
    plt.show()
    
    return


def plot_hist_predictions_list(W_and_H, Y, ndcg_val_list=None, len_max=1000):
    
    n_p = len(W_and_H)
    plt.figure()
    for ip in range(n_p):
        W, H = W_and_H[ip][0], W_and_H[ip][1]
        pred_data1 = pred_data_from_WH(W, H, Y, len_max)
        pred_data0 = pred_data_from_WH(W, H, sparse.csr_matrix(1-Y.toarray()), len_max)
        plt.subplot(1, n_p, ip+1)
        plt.hist(pred_data1, bins=50)
        plt.hist(pred_data0, bins=50)
        plt.legend(['ones', 'zeros'])
        if not(ndcg_val_list is None):
            plt.title(ndcg_val_list[ip]*100)
    plt.show()
    
    return


# Generate of list of user indexes for each batch
def user_idx_generator(n_users, batch_users):
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


# NDCG
def my_ndcg(true_ratings, pred_ratings, batch_users=5000, k=None, leftout_ratings=None):

    n_users, n_songs = true_ratings.shape
    pred_ratings_up = pred_ratings
    # Remove predictions on the left-out ratings ('train' for validation, and 'train+val' for testing)
    if leftout_ratings is not None:
        item_idx = np.zeros((n_users, n_songs), dtype=bool)
        item_idx[leftout_ratings.nonzero()] = True
        pred_ratings_up[item_idx] = -np.inf

    # Loop over user batches
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        # Take a batch
        true_ratings_batch = true_ratings[user_idx]
        pred_ratings_batch = pred_ratings_up[user_idx, :]
        # Call the NDCG for the current batch (depending on k)
        # If k not specified, compute the whole (standard) NDCG instead of its truncated version NDCG@k
        if k is None:
            ndcg_curr_batch = my_ndcg_batch(true_ratings_batch, pred_ratings_batch)
        else:
            ndcg_curr_batch = my_ndcg_k_batch(true_ratings_batch, pred_ratings_batch, k)
        res.append(ndcg_curr_batch)

    # Stack and get mean and std over users
    ndcg = np.hstack(res)
    # Remove 0s (artifically added to avoid warnings)
    ndcg[ndcg == 0] = np.nan
    ndcg_mean = np.nanmean(ndcg)
    ndcg_std = np.nanstd(ndcg)

    return ndcg_mean, ndcg_std


def my_ndcg_batch(true_ratings, pred_ratings):

    all_rank = np.argsort(np.argsort(-pred_ratings, axis=1), axis=1)

    # build the discount template
    tp = 1. / np.log2(np.arange(2, true_ratings.shape[1] + 2))
    all_disc = tp[all_rank]

    # Binarize the true ratings
    true_ratings_bin = (true_ratings > 0).tocoo()

    # Get the disc
    disc = sparse.csr_matrix((all_disc[true_ratings_bin.row, true_ratings_bin.col],
                              (true_ratings_bin.row, true_ratings_bin.col)),
                             shape=all_disc.shape)

    # DCG, ideal DCG and normalized DCG
    dcg = np.array(disc.sum(axis=1)).ravel()
    idcg = np.array([tp[:n].sum() for n in true_ratings.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg


def my_ndcg_k_batch(true_ratings, pred_ratings, k=100):

    n_users_currbatch = true_ratings.shape[0]
    idx_topk_part = bn.argpartition(-pred_ratings, k, axis=1)
    topk_part = pred_ratings[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(n_users_currbatch)[:, np.newaxis], idx_part]

    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (true_ratings[np.arange(n_users_currbatch)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum() for n in true_ratings.getnnz(axis=1)])
    ndcg = dcg / (idcg + 1e-8)

    return ndcg

# EOF

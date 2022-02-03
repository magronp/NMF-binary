#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from scipy import sparse
import os
from matplotlib import pyplot as plt
import pyreadr


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def build_split_masks(mat_shape, prop_train=0.5, prop_val=0.25):

    len_tot = np.prod(mat_shape)
    all_ind = np.zeros(len_tot)
    all_ind[:int(prop_train * len_tot)] = 2
    all_ind[int(prop_train * len_tot):int((prop_train + prop_val) * len_tot)] = 1
    np.random.shuffle(all_ind)
    mat_ind = np.reshape(all_ind, mat_shape)
    train_mask = (mat_ind == 2) * 1
    val_mask = (mat_ind == 1) * 1
    test_mask = (mat_ind == 0) * 1

    return train_mask, val_mask, test_mask


def get_perplexity(Y, Y_hat, mask=None, eps=1e-8):

    if mask is None:
        mask = np.ones_like(Y)

    perplx = Y * np.log(Y_hat + eps) + (1-Y) * np.log(1 - Y_hat + eps)
    perplx = - np.sum(mask * perplx) / np.count_nonzero(mask)

    return perplx


def get_density(my_dataset, data_dir):
    
    dataset_path = data_dir + my_dataset
    Y = pyreadr.read_r(dataset_path + '.rda')[my_dataset].to_numpy()
    dens = np.count_nonzero(Y)/np.prod(Y.shape)
    
    return dens


def nbmf_loss(Y, W, H, prior_alpha=1., prior_beta=1., mask=None, eps=1e-8):

    if mask is None:
        mask = np.ones_like(Y)
    Y_hat = np.dot(W.T, H)
    logllik = Y * np.log(Y_hat + eps) + (1-Y) * np.log(1 - Y_hat + eps)
    priorlik = (prior_alpha-1) * np.log(H + eps) + (prior_beta-1) * np.log(1 - H + eps)
    
    perplx = - (np.sum(mask * logllik) + np.sum(priorlik))/ np.count_nonzero(mask)

    return perplx


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


# EOF

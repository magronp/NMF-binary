#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import os
import pyreadr
import pandas as pd


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def load_data_and_mask(data_dir, my_dataset, prop_train=0.5, prop_val=0.25):

    dataset_path = data_dir + my_dataset

    if my_dataset == 'chilevotes':
        data, train_mask, val_mask, test_mask = prep_data_votes(data_dir=data_dir, prop_train=prop_train, prop_val=prop_val)
        pyreadr.write_rdata(dataset_path + '.rda', data, my_dataset)
    else:
        data = pyreadr.read_r(dataset_path + '.rda')[my_dataset]
        # Create train / val / test split in the form of binary masks (and record the test mask for evaluation)
        train_mask, val_mask, test_mask = build_split_masks(data.shape)

    np.savez(dataset_path + '_split.npz', train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data, train_mask, val_mask, test_mask


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


def prep_data_votes(data_dir='data/', u_len=500, v_len=500, prop_train=0.5, prop_val=0.25):

    # Load the data frame and add a column corresponding to the outcome
    df = pd.read_csv(data_dir + 'chile.csv.zip', compression="zip")
    df['selected_option'] = (df['option_a_sorted'] == df['selected']) * 1 + (df['option_b_sorted'] == df['selected']) * (-1)

    # Get a subset of unique users and indices
    unique_users = df['uuid'].unique()
    unique_votes = df['card_id'].unique()
    df_u = df.loc[df['uuid'].isin(unique_users[:u_len])]
    df_v = df_u.loc[df_u['card_id'].isin(unique_votes[:v_len])]

    # update the lists of unique users and votes after the first filtering
    unique_users = df_v['uuid'].unique()
    unique_votes = df_v['card_id'].unique()
    M, N = len(unique_users), len(unique_votes)

    # Create dictionaries to map user and vote IDs to integers
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_users))
    vote2id = dict((vid, i) for (i, vid) in enumerate(unique_votes))

    # Build the data matrix
    Y = np.zeros((M, N))
    for i in range(len(df_v)):
        uid = user2id[df_v.iloc[[i]]['uuid'].item()]
        vid = vote2id[df_v.iloc[[i]]['card_id'].item()]
        Y[uid, vid] = df_v.iloc[[i]]['selected_option'].item()

    # Remove rows (users) where there is no vote (not possible to make predictions otherwise)
    ind0_rows = np.sum(np.abs(Y), axis=1) < 6
    Y = np.delete(Y, ind0_rows, axis=0)

    # Same with columns (votes) for which there is no enough user
    ind0_cols = np.sum(np.abs(Y), axis=0) < 6
    Y = np.delete(Y, ind0_cols, axis=1)

    # Create masks for the split (be sure there's no empty row or col in the training mask)
    valid_train = False
    while not valid_train:
        maskinit = np.random.uniform(0, 1, Y.shape) * np.abs(Y)
        mask_train = ((maskinit > 0) * (maskinit <= prop_train)) * 1
        valid_train = np.sum(np.sum(mask_train, axis=1) == 0) == 0 and np.sum(np.sum(mask_train, axis=0) == 0) == 0

    mask_val = ((maskinit > prop_train) * (maskinit <= prop_train + prop_val)) * 1
    mask_test = (maskinit > prop_train + prop_val) * 1

    # Finally, transform the -1 in 0 for consistency
    Y[Y == -1] = 0

    # Store the data in a pd dataframe
    data = pd.DataFrame(Y)

    return data, mask_train, mask_val, mask_test


def get_perplexity(Y, Y_hat, mask=None, eps=1e-8):

    if mask is None:
        mask = np.ones_like(Y)

    perplx = Y * np.log(Y_hat + eps) + (1-Y) * np.log(1 - Y_hat + eps)
    perplx = - np.sum(mask * perplx) / np.count_nonzero(mask)

    return perplx


def nbmf_loss(Y, W, H, A, B, mask=None, eps=1e-8):

    if mask is None:
        mask = np.ones_like(Y)
        
    Y_hat = np.dot(W.T, H)
    logllik = Y * np.log(Y_hat + eps) + (1-Y) * np.log(1 - Y_hat + eps)
    priorlik = A * np.log(H + eps) + B * np.log(1 - H + eps)
    
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


# Generate of list of user indexes for each batch
def user_idx_generator(n_users, batch_users):
    for start in range(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


# EOF

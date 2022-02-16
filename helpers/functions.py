#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import os
import pyreadr


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def load_data_and_mask(data_dir, my_dataset, prop_train=0.5, prop_val=0.25):

    # Load the raw data
    dataset_path = data_dir + my_dataset
    data = pyreadr.read_r(dataset_path + '.rda')[my_dataset]

    # Create train / val / test split in the form of binary masks (and record the test mask for evaluation)
    train_mask, val_mask, test_mask = build_split_masks(data.shape)

    # Record the masks
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


def get_perplexity(Y, Y_hat, mask=None, eps=1e-8):

    if mask is None:
        mask = np.ones_like(Y)

    perplx = Y * np.log(Y_hat + eps) + (1-Y) * np.log(1 - Y_hat + eps)
    perplx = - np.sum(mask * perplx) / np.count_nonzero(mask)

    return perplx


# EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import get_perplexity
import pyreadr


def eval_data_model(dataset, model, data_dir='data/', out_dir='outputs/'):

    # Load the data
    dataset_path = data_dir + dataset
    Y = pyreadr.read_r(dataset_path + '.rda')[dataset].to_numpy()

    # Estimates on the test set
    dataset_output_dir = out_dir + dataset + '/'
    loader = np.load(dataset_output_dir + model + '_model.npz')
    Y_hat = loader['Y_hat']
    tot_time = loader['time']

    # Perplexity (first load the proper test mask)
    test_mask = np.load(dataset_path + '_split.npz')['test_mask']
    perplx = get_perplexity(Y, Y_hat, mask=test_mask)

    return perplx, tot_time


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Define general paths, all datasets and models
    data_dir = 'data/'
    out_dir = 'outputs/'
    models = ['NBMF-EM', 'NBMF-MM', 'logPCA']
    datasets = ['animals', 'paleo', 'lastfm']
    #datasets = ['animals']

    # Loop over datasets
    for dataset in datasets:
        print('--------- ' + dataset)
        for model in models:
            perplx, tot_time = eval_data_model(dataset, model, data_dir=data_dir, out_dir=out_dir)
            print(model, "-- Perplexity {:.2f} ---- Time {:.4f} ".format(perplx, tot_time))

# EOF

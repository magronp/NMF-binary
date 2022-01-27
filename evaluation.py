#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import get_perplexity
import pyreadr


def eval_data_model(my_dataset, model_name, data_dir='data/', out_dir='outputs/'):

    # Load the data
    dataset_path = data_dir + my_dataset
    Y = pyreadr.read_r(dataset_path + '.rda')[my_dataset].to_numpy()

    # Estimates on the test set
    dataset_output_dir = out_dir + my_dataset + '/'
    loader = np.load(dataset_output_dir + model_name + '_model.npz')
    W, H = loader['W'], loader['H']
    Y_hat = np.dot(W, H.T)

    # Perplexity (first load the proper test mask)
    test_mask = np.load(dataset_path + '_split.npz')['test_mask']
    perplx = get_perplexity(Y, Y_hat, mask=test_mask)

    return perplx


# Set random seed for reproducibility
np.random.seed(12345)

# General path
data_dir = 'data/'
out_dir = 'outputs/'

for my_dataset in ['animals', 'paleo', 'lastfm']:
    print('--- Dataset: ' + my_dataset)
    for model_name in ['bmf_noprior', 'bmf']:
        perplx = eval_data_model(my_dataset, model_name, data_dir=data_dir, out_dir=out_dir)
        print(model_name + ' :', perplx)

# EOF

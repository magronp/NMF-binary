#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import create_folder, get_perplexity
import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter


def train_lpca(Y, k=2, max_iter=100, mask_leftout=None, Wini=None, Hini=None):

    if mask_leftout is None:
        mask_leftout = np.zeros(Y.shape)

    # Initialize the factors
    if (Wini is None) or (Hini is None):
        m, n = Y.shape
        Wini = np.random.uniform(0, 1, (m, k))
        Hini = np.random.uniform(0, 1, (n, k))

    # Defining the R script and loading the instance in Python
    r = robjects.r
    r['source']('helpers/lpca.R')

    # Loading the function we have defined in R.
    logPCA_function_r = robjects.globalenv['logisticPCA']

    # Convert inputs to R objects
    with localconverter(robjects.default_converter + pandas2ri.converter):
        Y_r = robjects.conversion.py2rpy(Y)

    with localconverter(robjects.default_converter + numpy2ri.converter):
        mask_leftout_r = robjects.conversion.py2rpy(mask_leftout)
        Wini_r = robjects.conversion.py2rpy(Wini)
        Hini_r = robjects.conversion.py2rpy(Hini)

    # Apply the logistic PCA function in R
    W_r, H_r, Y_hat_r, tot_time_r, loss_r, iters_r = logPCA_function_r(Y_r, mask_leftout_r, k, max_iter, Wini_r, Hini_r)

    # Convert the output back to python
    with localconverter(robjects.default_converter + pandas2ri.converter):
        Y_hat = robjects.conversion.rpy2py(Y_hat_r)
        W = robjects.conversion.rpy2py(W_r)
        H = robjects.conversion.rpy2py(H_r)
        tot_time = robjects.conversion.rpy2py(tot_time_r).item()
        loss = robjects.conversion.rpy2py(loss_r)
        iters = robjects.conversion.rpy2py(iters_r).item()

    return W, H, Y_hat, tot_time, loss, iters


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # General path
    data_dir = 'data/'

    # Load the data
    my_dataset = 'animals'
    dataset_path = data_dir + my_dataset
    data = pyreadr.read_r(dataset_path + '.rda')[my_dataset]
    Y = data.to_numpy()

    # Hyperparameters
    max_iter = 200
    k = 2

    # Compute logistic PCA and display perplexity
    W, H, Y_hat, tot_time, loss, iters = train_lpca(data, k, max_iter)
    perplx = get_perplexity(Y, Y_hat)
    print('Perplexity on the test set:', perplx)

# EOF

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


def train_lpca(Y, k=2, max_iter=100, mask_leftout=None):

    if mask_leftout is None:
        mask_leftout = np.zeros(Y.shape)

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

    # Apply the logistic PCA function in R
    W_r, H_r, Y_hat_r = logPCA_function_r(Y_r, mask_leftout_r, k, max_iter)

    # Convert the output back to python
    with localconverter(robjects.default_converter + pandas2ri.converter):
        Y_hat = robjects.conversion.rpy2py(Y_hat_r)
        W = robjects.conversion.rpy2py(W_r)
        H = robjects.conversion.rpy2py(H_r)

    return W, H, Y_hat


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # General path
    data_dir = '../data/'
    out_dir = '../outputs/'

    # Load the data
    my_dataset = 'paleo'
    dataset_path = data_dir + my_dataset
    data = pyreadr.read_r(dataset_path + '.rda')[my_dataset]
    Y = data.to_numpy()

    # Define and create the output directory (if needed)
    dataset_output_dir = out_dir + my_dataset + '/'
    create_folder(dataset_output_dir)

    # Hyperparameters
    max_iter = 100
    k = 2

    # Compute logistic PCA and display perplexity
    W, H, Y_hat = train_lpca(data, k, max_iter)
    perplx = get_perplexity(Y, Y_hat)
    print('Perplexity on the test set:', perplx)

# EOF

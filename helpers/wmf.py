#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

'''
Part of this script is adapted from the WMF code, which is available at: https://github.com/dawenl/content_wmf
If you use it, please acknowledge it by citing the corresponding paper:
"D. Liang, M. Zhan, D. Ellis, Content-Aware Collaborative Music Recommendation Using Pre-trained Neural Networks,
Proc. of ISMIR 2015."
'''

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from helpers.functions import load_tp_data_as_binary_csr, plot_hist_predictions, my_ndcg


def get_confidence(playcounts, alpha=2.0, epsilon=1e-6):
    conf = playcounts.copy()
    conf.data = alpha * np.log(1 + conf.data / epsilon)
    return conf


def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, YTYpR, batch_size, m, f):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f))
    B_stack = np.empty((current_batch_size, f, f))

    for ib, k in enumerate(range(lo, hi)):
        s_u, i_u = get_row(S, k)

        Y_u = Y[i_u]  # exploit sparsity
        A = (s_u + 1).dot(Y_u)

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B

    X_stack = solve_sequential(A_stack, B_stack)

    return X_stack


def recompute_factors_batched(Y, S, lambda_reg, batch_size=10000, n_jobs=4):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)

    n_batches = int(np.ceil(m / float(batch_size)))

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, YTYpR, batch_size, m, f)
                                  for b in range(n_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new


def factorize_wmf(Y, n_factors, n_iters=10, lambda_W=1e-2, lambda_H=100, batch_size=10000,
              n_jobs=4, init_std=0.01, alpha_conf=2.0, eps_conf=1e-6, val_data=None):

    # Get the confidence from the input data
    conf = get_confidence(Y, alpha=alpha_conf, epsilon=eps_conf)

    # Get some parameters
    n_users, n_items = conf.shape
    confT = conf.T.tocsr()

    # Initialize WMF matrices
    W = np.random.randn(n_users, n_factors) * init_std
    H = None
    ndcg_val = []
    # Iterate updates
    for _ in tqdm(range(n_iters)):

        H = recompute_factors_batched(W, confT, lambda_H, batch_size=batch_size, n_jobs=n_jobs)
        W = recompute_factors_batched(H, conf, lambda_W, batch_size=batch_size, n_jobs=n_jobs)
        
        if not(val_data is None):
            ndcg_mean = my_ndcg(val_data, W.dot(H.T), batch_users=batch_size, k=50, leftout_ratings=Y)[0]
            ndcg_val.append(ndcg_mean)
        
    return W, H, ndcg_val

""" 
# Adapted to python 3 with Jit
from numba import jit
@jit
def _inner(W, H, rows, cols, dtype):
    n_ratings = rows.size
    n_components = W.shape[1]
    assert H.shape[1] == n_components
    data = np.empty(n_ratings, dtype=dtype)
    for i in range(n_ratings):
        data[i] = 0.0
        for j in range(n_components):
            data[i] += W[rows[i], j] * H[cols[i], j]
    return data
"""


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Define the parameters
    curr_dataset = 'tp_small/'
    data_dir = 'data/' + curr_dataset
    lambda_W, lambda_H  = 1e-2, 100
    n_factors = 8
    n_iters = 10
    eps = 1e-8
    
    #  Get the number of songs and users in the training dataset (leave 5% of the songs for out-of-matrix prediction)
    n_users = len(open(data_dir + 'unique_uid.txt').readlines())
    n_songs = len(open(data_dir + 'unique_sid.txt').readlines())

    # Load the training and validation data
    train_data = load_tp_data_as_binary_csr(data_dir + 'train.num.csv', shape=(n_users, n_songs))[0]
    val_data = load_tp_data_as_binary_csr(data_dir + 'val.num.csv', shape=(n_users, n_songs))[0]
    test_data = load_tp_data_as_binary_csr(data_dir + 'test.num.csv', shape=(n_users, n_songs))[0]
    left_out_data = val_data + test_data
    Y = train_data
    
    W, H, ndcg_val = factorize_wmf(Y, n_factors=n_factors, n_iters=n_iters, lambda_W=lambda_W,
                         lambda_H=lambda_H, batch_size=1000,
                         n_jobs=-1, init_std=0.01, alpha_conf=2.0, eps_conf=1e-6, val_data=val_data)

    np.savez('outputs/' + curr_dataset + '/wmf_model.npz', W=W, H=H, hyper_params=[lambda_W, lambda_H, n_factors])
    
    # Vizualization
    plot_hist_predictions(W, H, val_data, len_max=200)
    
    # ca semble approcher 1 correctement que sur le train, mais c'est pas évident en val/test
    # a quantifier 
    
# EOF



# EOF

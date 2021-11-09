#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from joblib import Parallel, delayed
from numba import jit
from helpers.data_feeder import load_tp_data_as_csr
import pandas as pd
from sklearn.preprocessing import scale

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'

'''
Part of this script is adapted from the WMF code, which is available at: https://github.com/dawenl/content_wmf
If you use it, please acknowledge it by citing the corresponding paper:
"D. Liang, M. Zhan, D. Ellis, Content-Aware Collaborative Music Recommendation Using Pre-trained Neural Networks,
Proc. of ISMIR 2015."
'''


def get_row(S, i):
    lo, hi = S.indptr[i], S.indptr[i + 1]
    return S.data[lo:hi], S.indices[lo:hi]


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, BZ, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

    for ib, k in enumerate(range(lo, hi)):
        s_u, i_u = get_row(S, k)

        Y_u = Y[i_u]  # exploit sparsity
        A = (s_u + 1).dot(Y_u)

        if BZ is not None:
            A += BZ[:, k]

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B

    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def recompute_factors_batched(Y, S, lambda_reg, B=None, Z=None, dtype='float32', batch_size=10000, n_jobs=4):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)
    if B is not None:
        BZ = lambda_reg * (Z.dot(B)).T
    else:
        BZ = None

    n_batches = int(np.ceil(m / float(batch_size)))

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, BZ, YTYpR,
                                                       batch_size, m, f, dtype)
                                  for b in range(n_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new


def factorize(conf, n_factors, Z=None, n_iters=10, lambda_W=1e-2, lambda_H=100, lambda_B=1e-2, dtype='float32',
              batch_size=10000, n_jobs=4):

    # Get some parameters
    n_users, n_items = conf.shape
    init_std = 0.01
    confT = conf.T.tocsr()

    # Initialize WMF matrices
    W = np.random.randn(n_users, n_factors).astype(dtype) * init_std
    H = None
    B, XTXpR = None, None
    if Z is not None:
        n_feats = Z.shape[1]
        B = np.random.randn(n_feats, n_factors).astype(dtype) * init_std
        R = np.eye(n_feats)
        R[n_feats - 1, n_feats - 1] = 0
        XTXpR = Z.T.dot(Z) + lambda_B * R

    # Iterate updates
    for i in range(n_iters):
        H = recompute_factors_batched(W, confT, lambda_H, B=B, Z=Z, dtype=dtype, batch_size=batch_size, n_jobs=n_jobs)
        W = recompute_factors_batched(H, conf, lambda_W, dtype=dtype, batch_size=batch_size, n_jobs=n_jobs)
        if Z is not None:
            B = np.linalg.solve(XTXpR, Z.T.dot(H))

    return W, H, B


# Adapted to python 3 with Jit
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


def pure_content_recom(params):

    #  Get the number of songs and users in the 'in-matrix' dataset (leave 5% of the songs for out-of-matrix)
    n_users = len(open(params['data_dir'] + 'unique_uid.txt').readlines())
    n_songs_in = int(0.95 * len(open(params['data_dir'] + 'unique_sid.txt').readlines()))

    # Load the in-matrix data for training
    train_data = load_tp_data_as_csr(params['data_dir'] + 'in.train.num.csv', shape=(n_users, n_songs_in))[0]
    vad_data = load_tp_data_as_csr(params['data_dir'] + 'in.vad.num.csv', shape=(n_users, n_songs_in))[0]
    test_data = load_tp_data_as_csr(params['data_dir'] + 'in.test.num.csv', shape=(n_users, n_songs_in))[0]
    in_data = train_data + test_data + vad_data

    # Load AVD for user profile preference learning features (and scale)
    in_avd = pd.read_csv(params['data_dir'] + 'in.avd.num.csv').to_numpy()
    in_avd = scale(np.delete(in_avd, 0, axis=1), axis=0)
    users_avd = in_data.dot(in_avd)

    # Load test songs AVD
    test_avd = pd.read_csv(params['data_dir'] + 'out.avd.num.csv').to_numpy()
    test_avd = test_avd[test_avd[:, 0].argsort()]
    test_avd = scale(np.delete(test_avd, 0, axis=1), axis=0)

    # Predicted ratings
    pred_ratings = np.dot(users_avd, test_avd.T)

    return pred_ratings


# EOF

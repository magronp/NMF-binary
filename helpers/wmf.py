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
from numba import jit
from tqdm import tqdm


def get_confidence(playcounts, alpha, epsilon):
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


def solve_batch(b, S, Y, YTYpR, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

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


def recompute_factors_batched(Y, S, lambda_reg, dtype='float32', batch_size=10000, n_jobs=4):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)

    n_batches = int(np.ceil(m / float(batch_size)))

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, YTYpR, batch_size, m, f, dtype)
                                  for b in range(n_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new


def factorize_wmf(Y, n_factors, n_iters=10, lambda_W=1e-2, lambda_H=100, dtype='float32', batch_size=10000,
              n_jobs=4, init_std=0.01, alpha_conf=2.0, eps_conf=1e-6):

    # Get the confidence from the input data
    conf = get_confidence(Y, alpha=alpha_conf, epsilon=eps_conf)

    # Get some parameters
    n_users, n_items = conf.shape
    confT = conf.T.tocsr()

    # Initialize WMF matrices
    W = np.random.randn(n_users, n_factors).astype(dtype) * init_std
    H = None

    # Iterate updates
    for _ in tqdm(range(n_iters)):

        H = recompute_factors_batched(W, confT, lambda_H, dtype=dtype, batch_size=batch_size, n_jobs=n_jobs)
        W = recompute_factors_batched(H, conf, lambda_W, dtype=dtype, batch_size=batch_size, n_jobs=n_jobs)

    return W, H


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

# EOF

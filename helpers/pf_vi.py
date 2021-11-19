#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

'''
This script largely relies on the PF code from Olivier Gouvert, which is available at: https://github.com/Oligou/dcPF/blob/master/model/pf.py
If you use it, please acknowledge it by citing the corresponding paper:
"O. Gouvert, T. Oberlin, C. Févotte "Recommendation from Raw Data with Adaptive Compound Poisson Factorization",
'''

# %% GIBBS FOR ZIPF

## Model
# W ~ Gamma(aphaW,betaW)    ## UxK xN
# H ~ Gamma(aphaH,betaH)    ## IxK xN
# c ~ Poisson(V*W*H)        ## UxIxK xN
# y = sum(c)                ## UxI

## Conditional
# W|H,V,c ~ Gamma(aphaW+sum(c), betaW+sum(H))
# c|W,H ~ Mult(y, W*H)

## Order: W,H,C
# %%
import numpy as np
from scipy import sparse, special
import os
import time
import pickle
import sys
from tqdm import tqdm


class pf():
    def __init__(self, K,
                 alphaW=1., alphaH=1., betaW=1., betaH=1.):
        self.K = K
        self.alphaW = alphaW
        self.alphaH = alphaH
        self.betaW = betaW
        self.betaH = betaH
        self.score = {}
        self.classname = 'pf'
        # Save arg
        self.saved_args_init = locals()
        del self.saved_args_init['self']

    def fit(self, Y,
            seed=None, init='rand',
            opt_hyper=[],
            precision=10 ** (-5), max_iter=10 ** 5, min_iter=0,
            save=True, save_dir='', prefix=None, suffix=None):

        self.seed = seed
        if isinstance(seed, int):
            np.random.seed(seed)
        self.init = init
        self.opt_hyper = opt_hyper
        self.precision = precision
        self.min_iter = min_iter
        self.max_iter = max_iter
        # Save
        self.save = save
        self.save_dir = save_dir
        self.filename = self.filename(prefix, suffix)
        # Save arg
        self.saved_args_fit = locals()
        del self.saved_args_fit['Y']
        del self.saved_args_fit['self']
        # Timer
        start_time = time.time()

        # CONSTANTS
        U, I = Y.shape
        s_y = Y.sum()
        elbo_cst = - np.sum(special.gammaln(Y.data + 1))
        self.Elbo = [-float("inf")]

        # INIT
        if init == 'pf_bin':
            model_init = pf(K=self.K,
                               alphaW=self.alphaW, alphaH=self.alphaH)
            model_init.fit(Y > 0,
                           precision=precision * 10, min_iter=min_iter // 10, max_iter=max_iter // 10,
                           save=True, save_dir=save_dir, prefix='INIT_pfbin_' + prefix)
            Ew = model_init.Ew
            Eh = model_init.Eh
            self.model_init = model_init
        else:
            Ew = np.random.gamma(1., 1., (U, self.K))
            Eh = np.random.gamma(1., 1., (I, self.K))
        s_wh = np.dot(np.sum(Ew, 0, keepdims=True), np.sum(Eh, 0, keepdims=True).T)[0, 0]
        Sw, Sh, elboLoc = self.q_Mult(Y, Ew, Eh)

        for n in tqdm(range(max_iter)):

            # Compound Poisson
            # Hyper parameter
            if np.isin('beta', opt_hyper):
                self.betaW = self.alphaW / Ew.mean(axis=1, keepdims=True)
                self.betaH = self.alphaH / Eh.mean(axis=1, keepdims=True)
            if np.isin('betaH', opt_hyper):
                self.betaH = self.alphaH / np.mean(Eh)
            # Global
            Ew, Elogw, elboW = q_Gamma(self.alphaW, Sw,
                                       self.betaW, np.sum(Eh, axis=0))
            Eh, Elogh, elboH = q_Gamma(self.alphaH, Sh,
                                       self.betaH, np.sum(Ew, axis=0))
            s_wh = np.dot(np.sum(Ew, 0, keepdims=True), np.sum(Eh, 0, keepdims=True).T)[0, 0]
            # Local
            Sw, Sh, elboLoc = self.q_Mult(Y, np.exp(Elogw), np.exp(Elogh))
            # Elbo
            elbo = elboLoc - s_wh + elboW + elboH + elbo_cst
            self.rate = (elbo - self.Elbo[-1]) / np.abs(self.Elbo[-1])
            if elbo < self.Elbo[-1]:
                self.Elbo.append(elbo)
                raise ValueError('Elbo diminue!')
            if np.isnan(elbo):
                raise ValueError('elbo NAN')
            elif self.rate < precision and n >= min_iter:
                self.Elbo.append(elbo)
                break
            self.Elbo.append(elbo)

        u, i = Y.nonzero()
        self.Ew = Ew.copy()
        self.Eh = Eh.copy()
        self.Elogw = Elogw.copy()
        self.Elogh = Elogh.copy()

        self.duration = time.time() - start_time

        # Save
        if self.save:
            self.save_model()

    def q_Mult(self, Y, W, H):
        # Product
        u, i = Y.nonzero()
        Ydata = Y.data
        s = np.sum(W[u, :] * H[i, :], 1)
        # Mult
        R = sparse.csr_matrix((Ydata / s, (u, i)), shape=Y.shape)  # UxI
        Sh = ((R.T).dot(W)) * H
        Sw = (R.dot(H)) * W
        elbo = np.sum(Ydata * np.log(s))
        return Sw, Sh, elbo

    def filename(self, prefix, suffix):
        if prefix is not None:
            prefix = prefix + '_'
        else:
            prefix = ''
        if suffix is not None:
            suffix = '_' + suffix
        else:
            suffix = ''
        return prefix + self.classname + \
               '_K%d' % (self.K) + \
               '_alpha%.2f_%.2f' % (self.alphaW, self.alphaH) + \
               '_beta%.2f_%.2f' % (self.betaW, self.betaH) + \
               '_opthyper_' + '_'.join(sorted(self.opt_hyper)) + \
               '_init_' + self.init + \
               '_tol%.1e' % (self.precision) + \
               '_iter%.1e_%.1e' % (self.min_iter, self.max_iter) + \
               '_seed' + str(self.seed) + suffix

    def save_model(self):
        with open(os.path.join(self.save_dir, self.filename), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def generate(self):
        pass

    def copy_attributes(self, oobj):
        self.__dict__ = oobj.__dict__.copy()


def stat_gamma(shape, rate):
    E = shape / rate
    dig_shape = special.digamma(shape)
    Elog = dig_shape - np.log(rate)
    entropy = shape - np.log(rate) + special.gammaln(shape) + (1 - shape) * dig_shape
    return E, Elog, entropy


def gamma_elbo(shape, rate, Ex, Elogx):
    return (shape - 1) * Elogx - rate * Ex + shape * np.log(rate) - special.gammaln(shape)


def q_Gamma(shape, _shape, rate, _rate):
    E, Elog, entropy = stat_gamma(shape + _shape, rate + _rate)
    elbo = gamma_elbo(shape, rate, E, Elog)
    elbo = elbo.sum() + entropy.sum()
    return E, Elog, elbo


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


# %%

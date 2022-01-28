#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from matplotlib import pyplot as plt

list_nfactors = [2, 4, 8, 16, 32]
list_alpha = np.linspace(1, 5, 5)
list_beta = list_alpha

# Load the validation results
val_pplx = np.load('outputs/paleo/bmf_model_val.npz', allow_pickle=True)['val_pplx']
nk, n_alpha, n_beta = val_pplx.shape


# Check the influence of K (in the case there's no prior)
plt.figure()
positions = np.arange(nk)
plt.xticks(positions, [str(k) for k in list_nfactors])
plt.plot(val_pplx[:, 0, 0])

# Check the influence of alpha and beta (for the optimal K)
ind_k_opt, _, _ = np.unravel_index(val_pplx.argmin(), val_pplx.shape)
plt.figure()
plt.imshow(val_pplx[ind_k_opt, :, :], aspect='auto')

xpositions = np.arange(n_alpha)
plt.xticks(xpositions, [str(k) for k in list_alpha])
ypositions = np.arange(n_beta)
plt.yticks(ypositions, [str(k) for k in list_beta])
plt.colorbar()
plt.show()


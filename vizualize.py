#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from matplotlib import pyplot as plt
import pyreadr


list_nfactors = [2, 4, 8, 16]
list_alpha = np.linspace(1, 2, 11)
list_beta = list_alpha

# Load the validation results
model_name = 'nbmf'
my_dataset = 'paleo'
val_pplx = np.load('outputs/' + my_dataset + '/' + model_name + '_model_val.npz', allow_pickle=True)['val_pplx']
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

xpositions = np.arange(n_beta)
plt.xticks(xpositions, [str(k) for k in list_beta ])
ypositions = np.arange(n_alpha)
plt.yticks(ypositions, [str(k) for k in list_alpha])
plt.colorbar()
plt.show()


# Check H on the lastfm dataset
my_dataset = 'lastfm'
data = pyreadr.read_r('data/' + my_dataset + '.rda')[my_dataset]
H_nbmf_noprior = np.load('outputs/' + my_dataset + '/nbmf_noprior_model.npz', allow_pickle=True)['H']
H_nbmf = np.load('outputs/' + my_dataset + '/nbmf_model.npz', allow_pickle=True)['H']
H_lpca = np.load('outputs/' + my_dataset + '/lpca_model.npz', allow_pickle=True)['H']

# Take an arbitrary subset of bands for vizualization
plot_range = np.concatenate((np.arange(120, 130), np.arange(184, 199)), axis=0)
labels_plot = np.array(data.columns)[plot_range]

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(H_nbmf[plot_range, :], aspect='auto', cmap='binary')
ypositions = np.arange(len(labels_plot))
plt.yticks(ypositions, labels_plot)
plt.xticks([])
plt.subplot(1, 2, 2)
plt.imshow(H_lpca[plot_range, :], aspect='auto', cmap='binary')
plt.yticks([])
plt.xticks([])
plt.tight_layout()



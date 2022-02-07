#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from matplotlib import pyplot as plt
import pyreadr

list_nfactors = [2, 4, 8, 16]
list_alpha = np.linspace(1, 20, 20)
list_beta = list_alpha

# All datasets
model_name = 'nbmf'
datasets = ['animals', 'paleo', 'lastfm']
datasets = ['animals']
n_datasets = len(datasets)

# Loop over datasets
for i_d, my_dataset in enumerate(datasets):
        
    # Load the validation results
    val_pplx = np.load('outputs/' + my_dataset + '/' + model_name + '_model_val.npz', allow_pickle=True)['val_pplx']
    nk, n_alpha, n_beta = val_pplx.shape
    
    # Get the optimal indices
    ind_k_opt, ind_alpha, ind_beta_opt = np.unravel_index(val_pplx.argmin(), val_pplx.shape)
    
    '''
    # Check the influence of K (for optimal alpha and beta)
    plt.figure()
    positions = np.arange(nk)
    plt.xticks(positions, [str(k) for k in list_nfactors])
    plt.plot(val_pplx[:, ind_alpha, ind_beta_opt])
    '''
    
    # Check the influence of alpha and beta (for the optimal K)
    plt.subplot(1, n_datasets, i_d+1)
    plt.imshow(val_pplx[ind_k_opt, :, :], aspect='auto', cmap='gray')
    plt.gca().invert_yaxis()
    xpositions = np.arange(n_beta)
    plt.xticks(xpositions, [str(int(k*10)/10) for k in list_beta ])
    plt.xlabel(r'$\beta$')
    if i_d==0:
        ypositions = np.arange(n_alpha)
        plt.yticks(ypositions, [str(int(k*10)/10) for k in list_alpha])
        plt.ylabel(r'$\alpha$')
    else:
        plt.yticks([])
    plt.title(my_dataset)
    
plt.show()
plt.tight_layout()

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



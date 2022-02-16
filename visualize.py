#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import load_data_and_mask
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# General paths and lists
# Define general paths, all datasets and models
data_dir = 'data/'
out_dir = 'outputs/'
models = ['NBMF-EM', 'NBMF-MM', 'logPCA']
datasets = ['animals', 'paleo', 'lastfm']
n_models, n_datasets = len(models), len(datasets)

"""
Plot 1: Check the validation results for the NBMF model
"""
plt.figure()
for id, my_dataset in enumerate(datasets):

    # Load the validation results
    val_loader = np.load(out_dir + my_dataset + '/NBMF-MM_val.npz', allow_pickle=True)
    val_pplx, val_hyperparams = val_loader['val_pplx'], val_loader['list_hyper']
    list_nfactors, list_alpha, list_beta = val_hyperparams[0], val_hyperparams[1], val_hyperparams[2]

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
    plt.subplot(1, n_datasets, id+1)
    plt.imshow(val_pplx[ind_k_opt, :, :], aspect='auto', cmap='gray')
    plt.gca().invert_yaxis()
    xpositions = np.arange(n_beta)
    plt.xticks(xpositions, [str(int(k*10)/10) for k in list_beta])
    plt.xlabel(r'$\beta$')
    if id==0:
        ypositions = np.arange(n_alpha)
        plt.yticks(ypositions, [str(int(k*10)/10) for k in list_alpha])
        plt.ylabel(r'$\alpha$')
    else:
        plt.yticks([])
    plt.title(my_dataset)

plt.show()
plt.tight_layout()


"""
Plot 2: Results on the test set with variable initialization
"""
n_init = 10
test_pplx_all = np.zeros((n_init, n_models, n_datasets))
test_time_all = np.zeros((n_init, n_models, n_datasets))

# Load the results
for ind_d, dataset in enumerate(datasets):
    dataset_output_dir = out_dir + dataset + '/'
    for ind_m, model in enumerate(models):
        test_loader = np.load(dataset_output_dir + model + '_test_init.npz', allow_pickle=True)
        test_pplx_all[:, ind_m, ind_d] = test_loader['test_pplx']
        test_time_all[:, ind_m, ind_d] = test_loader['test_time']

# Display perplexity
plt.figure()
for ind_d, dataset in enumerate(datasets):
    plt.subplot(1, n_datasets, ind_d + 1)
    plt.boxplot(test_pplx_all[:, :, ind_d], showfliers=False)
    plt.title(dataset)
    plt.xticks([1, 2, 3], models)
    if ind_d == 0:
        plt.ylabel('Perplexity')
plt.show()
plt.tight_layout()

# Print mean perplexity and time over random initializations
for ind_d, dataset in enumerate(datasets):
    print('--------- ' + dataset)
    for ind_m, model in enumerate(models):
        print(model, "-- Perplexity {:.2f} ---- Time {:.4f} ".format(np.mean(test_pplx_all[:, ind_m, ind_d]),
                                                                     np.mean(test_time_all[:, ind_m, ind_d])))

"""
Plot 3: Check H on the lastfm dataset for the logPCA and NBMF methods
"""
# Load H
dataset = 'lastfm'
data = load_data_and_mask(data_dir, dataset)[0]
H_nbmf = np.load(out_dir + dataset + '/NBMF-MM_model.npz', allow_pickle=True)['H']
H_lpca = np.load(out_dir + dataset + '/logPCA_model.npz', allow_pickle=True)['H']

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
plt.show()
plt.tight_layout()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import pyreadr
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

    # Check the influence of K (for optimal alpha and beta)
    """
    plt.figure()
    positions = np.arange(nk)
    plt.xticks(positions, [str(k) for k in list_nfactors])
    plt.plot(val_pplx[:, ind_alpha, ind_beta_opt])
    """

    # Check the influence of alpha and beta (for the optimal K)
    plt.subplot(1, n_datasets, id+1)
    plt.imshow(val_pplx[ind_k_opt, :, :], aspect='auto', cmap='gray')
    plt.gca().invert_yaxis()
    xpositions = np.arange(n_beta)[0::2]
    plt.xticks(xpositions, [str(int(k*10)/10) for k in list_beta][0::2])
    plt.xlabel(r'$\beta$', fontsize=16)
    if id == 0:
        ypositions = np.arange(n_alpha)[0::2]
        plt.yticks(ypositions, [str(int(k*10)/10) for k in list_alpha][0::2])
        plt.ylabel(r'$\alpha$', fontsize=16)
    else:
        plt.yticks([])
    plt.title(my_dataset, fontsize=16)

plt.show()
plt.tight_layout()


"""
Plot 2: Results on the test set with various random initializations
"""
n_init = 10
test_pplx_all = np.zeros((n_init, n_models, n_datasets))
test_time_all = np.zeros((n_init, n_models, n_datasets))
test_iter_all = np.zeros((n_init, n_models, n_datasets))

# Load the results
for ind_d, dataset in enumerate(datasets):
    dataset_output_dir = out_dir + dataset + '/'
    for ind_m, model in enumerate(models):
        test_loader = np.load(dataset_output_dir + model + '_test_init.npz', allow_pickle=True)
        test_pplx_all[:, ind_m, ind_d] = test_loader['test_pplx']
        test_time_all[:, ind_m, ind_d] = test_loader['test_time']
        test_iter_all[:, ind_m, ind_d] = test_loader['test_iter']

# Display perplexity
models_labs = ['NBMF-EM', 'NBMF-MM', 'logPCA']
fig2 = plt.figure()
xpos = [1, 2, 3]
width_bp = 0.75
for ind_d, dataset in enumerate(datasets):
    plt.subplot(1, n_datasets, ind_d + 1)
    plt.boxplot(test_pplx_all[:, :, ind_d], showfliers=False, positions=xpos, widths=[width_bp, width_bp, width_bp])
    plt.title(dataset, fontsize=16)
    plt.xticks(xpos, models_labs, fontsize=12, rotation=50)
    plt.yticks(fontsize=12)
    if ind_d == 0:
        plt.ylabel('Perplexity', fontsize=14)
plt.show()
plt.tight_layout()

# Print mean perplexity, time, and number of iterations, over random initializations
for ind_d, dataset in enumerate(datasets):
    print('--------- ' + dataset)
    for ind_m, model in enumerate(models):
        print(model, "-- Perplexity {:.2f} ---- Time {:.4f} ---- Iters {:.1f} ".format(np.mean(test_pplx_all[:, ind_m, ind_d]),
                                                                     np.mean(test_time_all[:, ind_m, ind_d]),
                                                                     np.mean(test_iter_all[:, ind_m, ind_d])))

"""
Plot 3: Check H on the lastfm dataset for the logPCA and NBMF methods
"""
# Load H
dataset = 'lastfm'
data = pyreadr.read_r(data_dir + dataset + '.rda')[dataset]
H_nbmf = np.load(out_dir + dataset + '/NBMF-MM_model.npz', allow_pickle=True)['H']
H_lpca = np.load(out_dir + dataset + '/logPCA_model.npz', allow_pickle=True)['H']

# Swap components for visualization
H_nbmf[:, [2, 3]] = H_nbmf[:, [3, 2]]
H_lpca[:, [2, 3]] = H_lpca[:, [3, 2]]

# Take an arbitrary subset of bands and reorganize songs for vizualization
plot_range = np.concatenate((np.arange(120, 130), np.arange(184, 199)), axis=0)
plot_range = plot_range[[4, 5, 1, 0, 3, 14, 15, 2, 6, 7, 8, 9, 11, 12, 17, 18, 19, 10, 13, 16, 20, 21, 22, 23, 24]]

labels_plot = np.array(data.columns)[plot_range]
H_nbmf_plot = H_nbmf[plot_range, :]
H_lpca_plot = H_lpca[plot_range, :]
ypositions = np.arange(len(labels_plot))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(H_nbmf_plot, aspect='auto', cmap='binary')
plt.yticks(ypositions, labels_plot, fontsize=14)
plt.xticks([])
plt.title('NBMF-MM', fontsize=16)
plt.subplot(1, 2, 2)
plt.imshow(H_lpca_plot, aspect='auto', cmap='binary')
plt.yticks([])
plt.xticks([])
plt.title('logPCA', fontsize=16)
plt.show()
plt.tight_layout()

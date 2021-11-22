#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import json
import os
import pandas as pd
from helpers.functions import create_folder
import pyreadr


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def print_sparsity(data_dir):

    data_frame = pd.read_csv(data_dir + 'data.csv')

    usercount, songcount = get_count(data_frame, 'uid'), get_count(data_frame, 'sid')
    sparsity_level = float(data_frame.shape[0]) / (usercount.shape[0] * songcount.shape[0])
    print("There are %d triplets from %d users and %d songs (sparsity level %.3f%%)"
          % (data_frame.shape[0], usercount.shape[0], songcount.shape[0], sparsity_level * 100))

    return


def preprocess_tp(path_tp='data/train_triplets.txt', data_dir='data/', min_uc=20, min_sc=50, min_c=5):

    # Load Taste Profile data
    tp_file = os.path.join(path_tp)
    tp = pd.read_table(tp_file, header=None, names=['uid', 'sid', 'count'])

    # Only keep ratings >= min_c, otherwise they're not informative.
    tp = tp[tp['count'] >= min_c]

    # Only keep the triplets for songs which were listened to by at least min_sc users, and at least min_sc times
    songcount = get_count(tp, 'sid')
    songcount_minsc = songcount[songcount['size'] >= min_sc]
    tp = tp[tp['sid'].isin(songcount_minsc['sid'])]

    usercount = get_count(tp, 'uid')
    usercount_minuc = usercount[usercount['size'] >= min_uc]
    tp = tp[tp['uid'].isin(usercount_minuc['uid'])]

    # Record the filtered TP data
    tp.to_csv(data_dir + 'data.csv', index=False)

    return


def record_ids_dict(data_dir='data/'):

    # Load the tp data
    data = pd.read_csv(data_dir + 'data.csv')

    # Get all users & songs in filtered taste profile, shuffle them, and map to integer indices
    unique_sid = pd.unique(data['sid'])
    n_songs = len(unique_sid)

    # Shuffle songs
    idx = np.random.permutation(np.arange(n_songs))
    unique_sid = unique_sid[idx]

    # Create a dictionary for mapping user/song unique ids to integers
    unique_uid = pd.unique(data['uid'])
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

    # Record dictionaries and lists of unique sid/uid
    with open(data_dir + 'unique_uid.txt', 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)

    with open(data_dir + 'unique_sid.txt', 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    with open(data_dir + 'user2id.json', 'w') as f:
        json.dump(user2id, f)

    with open(data_dir + 'song2id.json', 'w') as f:
        json.dump(song2id, f)

    return


def split_tp(data_dir='data/'):

    # Load filtered TP data
    data_frame = pd.read_csv(data_dir + 'data.csv')

    # Load the list of unique sids
    unique_sid = []
    with open(data_dir + '/unique_sid.txt', 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    # Select 5% songs for out-of-matrix prediction
    n_songs = len(unique_sid)
    out_sid = unique_sid[int(0.95 * n_songs):]

    # Pick out 10% of the rating for test
    n_ratings = data_frame.shape[0]
    test = np.random.choice(n_ratings, size=int(0.1 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    test_tp = data_frame[test_idx]
    train_tp = data_frame[~test_idx]

    # Pick out 10% of the (remaining) training rating as validation set
    n_ratings = train_tp.shape[0]
    vad = np.random.choice(n_ratings, size=int(0.1 * n_ratings), replace=False)
    val_idx = np.zeros(n_ratings, dtype=bool)
    val_idx[vad] = True
    val_tp = train_tp[val_idx]
    train_tp = train_tp[~val_idx]

    # Save the .csv
    test_tp.to_csv(data_dir + 'test.csv', index=False)
    train_tp.to_csv(data_dir + 'train.csv', index=False)
    val_tp.to_csv(data_dir + 'val.csv', index=False)

    return


def numerize_subset(subset_to_numerize, song2id, user2id, data_dir='data/'):

    data_frame = pd.read_csv(data_dir + subset_to_numerize + '.csv')
    uid = list(map(lambda x: user2id[x], data_frame['uid']))
    sid = list(map(lambda x: song2id[x], data_frame['sid']))
    data_frame['uid'] = uid
    data_frame['sid'] = sid
    data_frame.to_csv(data_dir + subset_to_numerize + '.num.csv', index=False)


def numerize_tp(data_dir='data/'):

    # Load the user and song to id mappings
    with open(data_dir + 'user2id.json', 'r') as f:
        user2id = json.load(f)

    with open(data_dir + 'song2id.json', 'r') as f:
        song2id = json.load(f)

    # Numerize all the TP subsets
    for subset_to_numerize in ['train', 'test', 'val']:
        numerize_subset(subset_to_numerize, song2id, user2id, data_dir=data_dir)

    return


def prepare_dataset(data_dir='data/'):

    # Record IDs lists and dictionaries
    print('Record song / user ids and dictionaries...')
    record_ids_dict(data_dir=data_dir)

    # Create train / validation / test / out of matrix split
    print('Create Train/Val/Test splits...')
    split_tp(data_dir=data_dir)

    # Numerize the song and users id (to have matrix-like entry indices)
    print('Numerize the songs/users ID and record the data...')
    numerize_tp(data_dir=data_dir)

    return


def preprocess_rda(rda_type, path_rda='data/', data_dir='data'):

    # Create the folder where to store the data (if needed)
    create_folder(data_dir)

    data_frame = pyreadr.read_r(path_rda + rda_type + '.rda')[rda_type]
    if rda_type == 'lastfm':
        data_frame.insert(0, 'uid', ['u_' + str(i) for i in range(data_frame.shape[0])])
    else:
        data_frame.insert(0, 'uid', data_frame.index)
    col_names = data_frame.columns
    data_frame = data_frame.melt(id_vars='uid', value_vars=col_names)
    data_frame = data_frame.rename(columns={'variable': 'sid', 'value': 'count'})
    data_frame = data_frame[data_frame['count'] == 1]

    # Record the prepared Last FM data
    data_frame.to_csv(data_dir + 'data.csv', index=False)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    # Create the TP dataset
    """ 
    data_dir = 'data/tp_small/'
    create_folder(data_dir)
    # Filter out TP data
    min_uc, min_sc, min_c = 20, 50, 20
    preprocess_tp(path_tp='data/train_triplets.txt', data_dir=data_dir, min_uc=min_uc, min_sc=min_sc, min_c=min_c)
    # Prepare the whole dataset
    prepare_dataset(data_dir=data_dir)
    print_sparsity(data_dir)
    

    # Create the TP dataset
    data_dir = 'data/tp_med/'
    create_folder(data_dir)
    # Filter out TP data
    min_uc, min_sc, min_c = 20, 50, 10
    preprocess_tp(path_tp='data/train_triplets.txt', data_dir=data_dir, min_uc=min_uc, min_sc=min_sc, min_c=min_c)
    # Prepare the whole dataset
    prepare_dataset(data_dir=data_dir)
    print_sparsity(data_dir)
    """

    # Create the TP dataset
    data_dir = 'data/tp_big/'
    create_folder(data_dir)
    # Filter out TP data
    min_uc, min_sc, min_c = 20, 50, 7
    preprocess_tp(path_tp='data/train_triplets.txt', data_dir=data_dir, min_uc=min_uc, min_sc=min_sc, min_c=min_c)
    # Prepare the whole dataset
    prepare_dataset(data_dir=data_dir)
    print_sparsity(data_dir)

""" 
    # Create the other datasets (using RDA files)
    for rda_type in ['lastfm', 'animals', 'paleo']:
        data_dir = 'data/' + rda_type + '/'
        create_folder(data_dir)
        preprocess_rda(rda_type, path_rda='data/', data_dir=data_dir)
        # Prepare the whole dataset
        prepare_dataset(data_dir=data_dir)
        print_sparsity(data_dir)
"""
# EOF

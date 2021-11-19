#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import json
import os
import pandas as pd


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def load_tp(data_dir='data/'):

    # Load Taste Profile data
    tp_file = os.path.join(data_dir, 'train_triplets.txt')
    tp = pd.read_table(tp_file, header=None, names=['uid', 'sid', 'count'])

    return tp


def filter_tp_inactive(tp, min_uc=20, min_sc=50, min_c=5, verbose=True):

    # Only keep ratings >= min_c, otherwise they're not informative.
    tp = tp[tp['count'] >= min_c]

    # Only keep the triplets for songs which were listened to by at least min_sc users, and at least min_sc times
    songcount = get_count(tp, 'sid')
    songcount_minsc = songcount[songcount['size'] >= min_sc]
    tp = tp[tp['sid'].isin(songcount_minsc['sid'])]

    usercount = get_count(tp, 'uid')
    usercount_minuc = usercount[usercount['size'] >= min_uc]
    tp = tp[tp['uid'].isin(usercount_minuc['uid'])]

    # Update both usercount and songcount after filtering
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')

    # Print the sparsity level
    if verbose:
        sparsity_level = float(tp.shape[0]) / (usercount.shape[0] * songcount.shape[0])
        print("After filtering, there are %d triplets from %d users and %d songs (sparsity level %.3f%%)"
              % (tp.shape[0], usercount.shape[0], songcount.shape[0], sparsity_level*100))

    return tp, usercount, songcount


def record_tp_data(tp, data_dir='data/'):

    # Get all users & songs in filtered taste profile, shuffle them, and map to integer indices
    unique_sid = pd.unique(tp['sid'])
    n_songs = len(unique_sid)

    # Shuffle songs
    idx = np.random.permutation(np.arange(n_songs))
    unique_sid = unique_sid[idx]

    # Create a dictionary for mapping user/song unique ids to integers
    unique_uid = pd.unique(tp['uid'])
    user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
    song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))

    # Record TP data, dictionaries and lists of unique sid/uid
    tp.to_csv('data/tp.csv', index=False)

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

    return unique_sid


def split_tp(tp=None, unique_sid=None, data_dir='data/'):

    # Taste profile data, if not provided as an input
    if tp is None:
        tp = pd.read_csv(data_dir + 'tp.csv')

    # List of unique songs if not provided as an input
    if unique_sid is None:
        unique_sid = []
        with open(data_dir + '/unique_sid.txt', 'r') as f:
            for line in f:
                unique_sid.append(line.strip())

    # Select 5% songs for out-of-matrix prediction
    n_songs = len(unique_sid)
    out_sid = unique_sid[int(0.95 * n_songs):]

    # Generate in and out of matrix split from TP
    out_tp = tp[tp['sid'].isin(out_sid)]
    in_tp = tp[~tp['sid'].isin(out_sid)]

    # Pick out 10% of the rating for test
    n_ratings = in_tp.shape[0]
    test = np.random.choice(n_ratings, size=int(0.1 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    test_tp = in_tp[test_idx]
    train_tp = in_tp[~test_idx]

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

    data_tp = pd.read_csv(data_dir + subset_to_numerize + '.csv')
    uid = list(map(lambda x: user2id[x], data_tp['uid']))
    sid = list(map(lambda x: song2id[x], data_tp['sid']))
    data_tp['uid'] = uid
    data_tp['sid'] = sid
    data_tp.to_csv(data_dir + subset_to_numerize + '.num.csv', index=False)


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


def prepare_whole_tp(min_uc=20, min_sc=50, min_c=5, data_dir='data/', verbose=True):

    # Load the TP data and filter out bad audio / inactive data
    if verbose:
        print('Load Taste Profile data and filter out bad audio...')
    tp = load_tp(data_dir)
    tp, usercount, songcount = filter_tp_inactive(tp, min_uc=min_uc, min_sc=min_sc, min_c=min_c, verbose=verbose)

    # Record clean TP data and dictionaries
    if verbose:
        print('Record Taste Profile data...')
    unique_sid = record_tp_data(tp, data_dir=data_dir)

    # Create train / validation / test / out of matrix split
    if verbose:
        print('Create Train/Val/Test splits...')
    split_tp(tp, unique_sid, data_dir=data_dir)

    # Numerize the song and users id (to have matrix-like entry indices)
    if verbose:
        print('Numerize the songs/users ID and record the data...')
    numerize_tp(data_dir=data_dir)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    MIN_USER_COUNT, MIN_SONG_COUNT, MIN_COUNT = 20, 70, 20
    data_dir = 'data/'

    prepare_whole_tp(min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT, min_c=MIN_COUNT, data_dir='data/', verbose=True)

# EOF

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sqlite3
import numpy as np
import pandas as pd

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'

'''
This script is adapted from the WMF code, which is available at: https://github.com/dawenl/content_wmf
If you use it, please acknowledge it by citing the corresponding paper:
"D. Liang, M. Zhan, D. Ellis, Content-Aware Collaborative Music Recommendation Using Pre-trained Neural Networks,
Proc. of ISMIR 2015."
'''


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def load_tp(data_dir='data/'):

    # Load Taste Profile data
    tp_file = os.path.join(data_dir, 'train_triplets.txt')
    tp = pd.read_table(tp_file, header=None, names=['uid', 'sid', 'count'])

    return tp


def filter_tp_inactive(tp, min_uc=20, min_sc=50, min_c=5):

    # Only keep ratings >= 5, otherwise they're not informative.
    tp = tp[tp['count'] >= min_c]
    '''
    # Keep top users (based on user counts)
    usercount = get_count(tp, 'uid')
    unique_uid = usercount.index
    p_users = usercount / usercount.sum()
    idx = np.random.choice(len(unique_uid), size=n_users, replace=False, p=p_users.tolist())
    unique_uid = unique_uid[idx]
    tp = tp[tp['uid'].isin(unique_uid)]

    # Keep top songs (based on songs counts)
    songcount = get_count(tp, 'sid')
    unique_sid = songcount.index
    p_songs = songcount / songcount.sum()
    idx = np.random.choice(len(unique_sid), size=n_songs, replace=False, p=p_songs.tolist())
    unique_sid = unique_sid[idx]
    tp = tp[tp['sid'].isin(unique_sid)]
    '''

    # Only keep the triplets for songs which were listened to by at least min_sc users, and at least min_sc times
    songcount = get_count(tp, 'sid')
    tp = tp[tp['sid'].isin(songcount.index[songcount >= min_sc])]
    usercount = get_count(tp, 'uid')
    tp = tp[tp['uid'].isin(usercount.index[usercount >= min_uc])]


    # Update both usercount and songcount after filtering
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')
    # Print the sparsity level
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

    # Pick out 20% of the rating for in-matrix prediction
    n_ratings = in_tp.shape[0]
    test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True
    test_tp = in_tp[test_idx]
    train_tp = in_tp[~test_idx]

    # Pick out 10% of the (remaining) training rating as validation set
    n_ratings = train_tp.shape[0]
    vad = np.random.choice(n_ratings, size=int(0.10 * n_ratings), replace=False)
    vad_idx = np.zeros(n_ratings, dtype=bool)
    vad_idx[vad] = True
    vad_tp = train_tp[vad_idx]
    train_tp = train_tp[~vad_idx]

    # Save the .csv
    test_tp.to_csv(data_dir + 'in.test.csv', index=False)
    train_tp.to_csv(data_dir + 'in.train.csv', index=False)
    vad_tp.to_csv(data_dir + 'in.vad.csv', index=False)
    out_tp.to_csv(data_dir + 'out.test.csv', index=False)

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
    for subset_to_numerize in ['in.train', 'in.test', 'in.vad', 'out.test']:
        numerize_subset(subset_to_numerize, song2id, user2id, data_dir=data_dir)

    return


def prepare_whole_tp(min_uc=20, min_sc=50, min_c=5, data_dir='data/'):

    # Load the TP data and filter out bad audio / inactive data
    tp = load_tp(data_dir)
    tp, usercount, songcount = filter_tp_inactive(tp, min_uc=min_uc, min_sc=min_sc, min_c=min_c)

    # Record clean TP data and dictionaries
    unique_sid = record_tp_data(tp, data_dir=data_dir)

    # Create train / validation / test / out of matrix split
    split_tp(tp, unique_sid, data_dir=data_dir)

    # Numerize the song and users id (to have matrix-like entry indices)
    numerize_tp(data_dir=data_dir)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    MIN_USER_COUNT, MIN_SONG_COUNT, MIN_COUNT = 20, 50, 5
    data_dir = 'data/'

    prepare_whole_tp(min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT, min_c=MIN_COUNT, data_dir=data_dir)


# EOF

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
from helpers import tasteprofile
from helpers.extract_features import handle_features_whole_dataset

__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'


def filter_tp_essentia(tp, data_dir='data/'):

    # Get the list of sid in the ESSENTIA extracted features
    essentia_features = pd.read_csv(data_dir + 'msd_played_songs_essentia.csv', skipinitialspace=True, usecols=[0])
    essentia_sid = [ll.replace('.mp3', '') for ll in essentia_features['metadata_tags_file_name']]
    # Keep only TP data whose sid match
    tp = tp[tp['sid'].isin(essentia_sid)]

    return tp


def record_features_split(features_name='avd_full', unique_sid=None, data_dir='data/'):
    # Get the list of songs ID if not provided
    if unique_sid is None:
        unique_sid = []
        with open(data_dir + '/unique_sid.txt', 'r') as f:
            for line in f:
                unique_sid.append(line.strip())

    # First, load the whole set of features
    my_features = pd.read_csv('data/' + features_name + '.csv')

    # Only keep features whose SID is in the list of sid of the dataset
    my_features = my_features[my_features['sid'].isin(unique_sid)]

    # Create in / out of matrix split for the AVD
    n_songs = len(unique_sid)
    out_sid = unique_sid[int(0.95 * n_songs):]
    out_features = my_features[my_features['sid'].isin(out_sid)]
    in_features = my_features[~my_features['sid'].isin(out_sid)]

    # Record
    in_features.to_csv('data/in.' + features_name + '.csv', index=False)
    out_features.to_csv('data/out.' + features_name + '.csv', index=False)

    # Numerize and record
    with open('data/song2id.json', 'r') as f:
        song2id = json.load(f)

    sid = list(map(lambda x: song2id[x], in_features['sid']))
    in_features = in_features.assign(sid=sid)
    in_features.to_csv('data/in.' + features_name + '.num.csv', index=False)

    sid = list(map(lambda x: song2id[x], out_features['sid']))
    out_features = out_features.assign(sid=sid)
    out_features.to_csv('data/out.' + features_name + '.num.csv', index=False)


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(12345)

    MIN_USER_COUNT, MIN_SONG_COUNT, MIN_COUNT = 20, 50, 5
    data_dir = 'data/'

    # Load the TP data and filter out bad audio / non-ESSENTIA songs / inactive data
    print('Load Taste Profile data and filter non-ESSENTIA songs and bad audio...')
    tp = tasteprofile.load_tp(data_dir)
    tp = filter_tp_essentia(tp, data_dir=data_dir)
    tp = tasteprofile.filter_tp_inactive(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT, min_c=MIN_COUNT)[0]

    # Record clean TP data and dictionaries
    print('Record Taste Profile data...')
    unique_sid = tasteprofile.record_tp_data(tp, data_dir=data_dir)

    # Create train / validation / test / out of matrix split
    print('Create In/Out and Train/Val/Test splits...')
    tasteprofile.split_tp(tp, unique_sid, data_dir=data_dir)

    # Numerize the song and users id (to have matrix-like entry indices)
    print('Numerize the songs/users ID and record the data...')
    tasteprofile.numerize_tp(data_dir=data_dir)

    # Compute and record ESSENTIA/AVD/factor loadings using
    print('Extract AVD features...')
    handle_features_whole_dataset(unique_sid=None, data_dir=data_dir)

    # Split all the features for in/out of matrix and record
    print('Split the features (in/out-matrix) and record...')
    record_features_split(features_name='avd', unique_sid=unique_sid, data_dir=data_dir)
    record_features_split(features_name='essentia', unique_sid=unique_sid, data_dir=data_dir)

# EOF


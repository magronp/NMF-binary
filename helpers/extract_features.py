#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.preprocessing import scale
from factor_analyzer import FactorAnalyzer
import sqlite3

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def load_essentia(unique_sid=None, data_dir='data/'):

    # Load the list of relevent ESSENTIA features for computing AVD
    rel_features_list = pd.read_csv(data_dir + 'essentia_varnames.csv')
    rel_features = list(rel_features_list['colname'])

    # Load the ESSENTIA computed features and change the sid (to make it similar to the list of unique sids)
    essentia_features = pd.read_csv(data_dir + 'msd_played_songs_essentia.csv')
    essentia_features.rename(columns={'metadata_tags_file_name': 'sid'}, inplace=True)
    essentia_features['sid'] = [ll.replace('.mp3', '') for ll in essentia_features['sid']]

    # Remove the songs not in TP
    if not (unique_sid is None):
        essentia_features = essentia_features[essentia_features['sid'].isin(unique_sid)]

    # Final unique list of SID
    essentia_sid = list(essentia_features['sid'])
    essentia_features = essentia_features[rel_features]

    return essentia_features, essentia_sid


def extract_avd(essentia_features, essentia_sid, n_factors=3):

    # List relevant ESSENTIA features and AVD factors
    rel_features = list(essentia_features.columns)
    avd_tags = ['Arousal', 'Valence', 'Depth']

    # First scale the features to 0 mean and unit variance
    essentia_scaled = scale(essentia_features.to_numpy(), axis=0)

    # PCA
    my_fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin')
    my_fa.fit(essentia_scaled)

    # Components loadings (for further analysis)
    avd_loadings = pd.DataFrame(my_fa.loadings_, columns=avd_tags)
    avd_loadings.insert(0, 'Feature', rel_features)

    # Data frame containing AVD
    avd = pd.DataFrame(my_fa.transform(essentia_scaled), columns=avd_tags)
    avd.insert(0, 'sid', essentia_sid)

    # Data frame containing Essentia features (plus sid)
    essentia_features.insert(0, 'sid', essentia_sid)

    return avd, essentia_features, avd_loadings


def handle_features_whole_dataset(unique_sid=None, data_dir='data/'):

    # Load Essentia features
    essentia_features, essentia_sid = load_essentia(unique_sid=unique_sid, data_dir=data_dir)

    # Calculate the AVD factors
    avd, essentia_features, avd_loadings = extract_avd(essentia_features, essentia_sid, n_factors=3)

    # Record Essentia and AVD features, as well as factor loadings
    essentia_features.to_csv(data_dir + 'essentia.csv', index=False)
    avd.to_csv(data_dir + 'avd.csv', index=False)
    avd_loadings.to_csv(data_dir + 'loadings.csv', index=False)

    return


def get_song_info_from_sid(conn, sid):
    cur = conn.cursor()
    cur.execute("SELECT title, artist_name FROM songs WHERE song_id = '%s'" % (sid))
    title, artist = cur.fetchone()
    return title, artist


def get_song_title_artist(md_dbfile, sid_list):

    list_songs = []
    with sqlite3.connect(os.path.join(md_dbfile)) as conn:
        for sid in list(sid_list):
            cur = conn.cursor()
            cur.execute("SELECT title, artist_name FROM songs WHERE song_id = '%s'" % sid)
            title, artist = cur.fetchone()
            list_songs.append(title + ' BY ' + artist)

    return list_songs


def get_song_top_avd(path_avd, data_dir='data/', top_songs=5, min_values=False):

    avd_tags = ['Arousal', 'Valence', 'Depth']
    md_dbfile = os.path.join(data_dir, 'track_metadata.db')
    avd = pd.read_csv(path_avd)

    avd_top = pd.DataFrame(columns=avd_tags)

    for feat in avd_tags:
        idx_top = avd[feat].sort_values(ascending=min_values)[:top_songs].index
        sid_top = avd['sid'].to_numpy()[idx_top]
        song_info = get_song_title_artist(md_dbfile, sid_top)
        avd_top[feat] = song_info

    return avd_top


if __name__ == '__main__':

    data_dir = 'data/'

    # Compute and record AVD/Essentia/loadings using all the available data
    handle_features_whole_dataset(unique_sid=None, data_dir='data/')

    # Visualize songs corresponding to min/max value for the different AVDs
    top_songs = 5
    avd_sub_dir = data_dir + 'in.avd.csv'
    topsongs_avd_sub = get_song_top_avd(avd_sub_dir, data_dir, top_songs=top_songs, min_values=False)
    minsongs_avd_sub = get_song_top_avd(avd_sub_dir, data_dir, top_songs=top_songs, min_values=True)

    # Display the factor loadings
    print(pd.read_csv(data_dir + 'loadings.csv'))

# EOF

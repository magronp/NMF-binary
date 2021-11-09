#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.sparse

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def get_confidence(playcounts, alpha, epsilon):
    conf = playcounts.copy()
    conf.data = alpha * np.log(1 + conf.data / epsilon)
    return conf


def load_tp_data_as_csr(csv_file, shape):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    sparse_tp = scipy.sparse.csr_matrix((count, (rows, cols)), dtype=np.int16, shape=shape)
    # Binarize the data
    sparse_tp.data = np.ones_like(sparse_tp.data)
    return sparse_tp, rows, cols

# EOF

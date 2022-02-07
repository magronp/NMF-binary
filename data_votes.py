#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:36:05 2022

@author: paul
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.functions import create_folder, get_perplexity, build_split_masks
from helpers.nbmf import train_nbmf
from helpers.lpca import train_lpca
import pyreadr
import pandas as pd

df = pd.read_csv("data/chile.csv.zip", compression="zip")
df['option_a_selected'] = df['option_a_sorted'] == df['selected']
df['option_b_selected'] = df['option_b_sorted'] == df['selected']

unique_users = df['uuid'].unique()
unique_votes = df['card_id'].unique()


M, N = 400, 400

selected_users = unique_users[:M]
selected_votes = unique_votes[:N]

df.loc[df['uuid'].isin(selected_users)]


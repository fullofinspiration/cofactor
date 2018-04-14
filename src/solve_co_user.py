import itertools
import glob
import os
import sys
from joblib import Parallel, delayed

DATA_DIR = 'E:\datasets\ml-100k\pro2'

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from scipy import sparse
import seaborn as sns

sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')
import sys


def _coord_batch(lo, hi, train_data):
    rows = []
    cols = []
    for u in xrange(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
    np.save(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
    pass


def use_coord_batch(start_idx, end_idx, train_data):
    #Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data) for lo, hi in zip(start_idx, end_idx))
    for lo, hi in zip(start_idx, end_idx):
        _coord_batch(lo, hi, train_data)

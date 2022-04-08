import os
import obonet

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp_sparse
import scanorama
import scanpy as sc
import scvi
import seaborn as sns
import string


import alluvial

from OnClass.OnClassModel import OnClassModel
import logging

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.backends.backend_pdf
from numba import boolean, float32, float64, int32, int64, vectorize
from collections import defaultdict

from .utils import *


def absolute_accuracy(adata, pred_key, gt_key, save_key=None):
    pred = adata.obs[pred_key].str.lower()
    gt = adata.obs[gt_key].str.lower()

    acc = (pred==gt).astype(int)
    if save_key is not None:
        adata.obs[save_key] = acc
    return acc
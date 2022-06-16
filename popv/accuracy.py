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

def ontology_accuracy(adata, pred_key, gt_key, obofile, save_key=None):
    G = make_ontology_dag(obofile).reverse()
    nodes = set(G.nodes())
    if not save_key:
        save_key = "ontology_accuracy"
    adata.obs[save_key] = "na"
    def match_type(n1, n2):
        if n1 == n2:
            return "exact"
        elif not set(G.predecessors(n1)).isdisjoint(G.predecessors(n2)):
            return "sibling"
        elif n1 in set(G.predecessors(n2)):
            return "parent"
        elif n2 in set(G.predecessors(n1)):
            return "child"
        else:
            return "no match"
    adata.obs[save_key] = adata.obs.apply(lambda x : match_type(x[pred_key], x[gt_key]), axis=1)
    return adata.obs[save_key]

import logging
import os
import string
from collections import defaultdict

import anndata
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import obonet
import pandas as pd
import scanorama
import scanpy as sc
import scipy.sparse as sp_sparse
import scvi
import seaborn as sns
from numba import boolean, float32, float64, int32, int64, vectorize
from OnClass.OnClassModel import OnClassModel
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from . import alluvial
from .utils import *


def sample_report(adata, cell_type_key, score_key, pred_keys):
    adata.obs["counts"] = np.zeros(len(adata.obs))
    _counts_adata = (
        adata.obs.groupby([cell_type_key, score_key]).count()[["counts"]].reset_index()
    )
    counts_adata = _counts_adata.pivot(cell_type_key, score_key, "counts")
    counts_adata = counts_adata.dropna()
    np_counts = counts_adata.dropna().to_numpy()
    row_sums = np_counts.sum(axis=1)
    new_matrix = np_counts / row_sums[:, np.newaxis]
    ax = (
        pd.DataFrame(
            data=new_matrix, index=counts_adata.index, columns=counts_adata.columns
        )
        .sort_values(7)
        .plot(kind="bar", stacked=True, figsize=(20, 7))
    )
    plt.title("Agreement per celltype", fontsize=16)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
    abundance_adata = adata.obs.groupby(cell_type_key).count()[["counts"]]
    abundance_adata = abundance_adata.dropna()
    ax = abundance_adata.plot(kind="bar", figsize=(20, 7))
    plt.title("Cell type abundance plot", fontsize=16)
    ax.bar_label(ax.containers[0])
    plt.show()
    for key in pred_keys:
        counts_adata = (
            adata.obs.groupby([key, cell_type_key])
            .count()
            .reset_index()
            .pivot(key, cell_type_key, "counts")
        )

        np_counts = counts_adata.dropna().to_numpy()
        row_sums = np_counts.sum(axis=0)
        new_matrix = np_counts / row_sums[np.newaxis, :]
        new_index = [
            counts_adata.index[r] + " " + str(np.sum(np_counts[r]))
            for r in range(new_matrix.shape[0])
        ]
        new_columns = [
            counts_adata.columns[c] + " " + str(np.sum(np_counts[:, c]))
            for c in range(new_matrix.shape[1])
        ]
        input_data = pd.DataFrame(
            data=new_matrix, index=new_index, columns=new_columns
        ).to_dict()
        cmap = matplotlib.cm.get_cmap("jet")
        sorted_index = np.array(new_index)[
            sorted(list(range(new_matrix.shape[0])), key=lambda r: np.sum(np_counts[r]))
        ]
        sorted_columns = np.array(new_columns)[
            sorted(
                list(range(new_matrix.shape[1])), key=lambda c: np.sum(np_counts[:, c])
            )
        ]
        ax = alluvial.plot(
            input_data,
            alpha=0.4,
            color_side=1,
            figsize=(5, 10),
            wdisp_sep=" " * 2,
            cmap=cmap,
            fontname="Monospace",
            label_shift=2,
            b_sort=list(sorted_index),
            a_sort=list(sorted_columns),
            labels=("Method", "Consensus"),
        )

        ax.set_title(key, fontsize=14, fontname="Monospace")
        plt.show()


def make_agreement_plots(adata, methods, save_folder):
    # TODO should this be pulling from resultsadata?

    # clear all existing figures first
    # or else this will interfere with the pdf saving capabilities
    fig_nums = plt.get_fignums()
    for num in fig_nums:
        plt.close(num)

    for method in methods:
        print("Making confusion matrix for {}".format(method))
        x_label = method
        y_label = "consensus_prediction"
        prediction_eval(
            adata.obs[x_label],
            adata.obs[y_label],
            name=method,
            x_label=x_label,
            y_label=y_label,
            res_dir=save_folder,
        )
    plt.close()

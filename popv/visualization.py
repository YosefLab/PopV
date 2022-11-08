"""Function to plot cell-type annotation."""

import logging
import os
from typing import Optional

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import _alluvial


def _sample_report(adata, cell_type_key, score_key, pred_keys):
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
            sorted(range(new_matrix.shape[0]), key=lambda r: np.sum(np_counts[r]))
        ]
        sorted_columns = np.array(new_columns)[
            sorted(range(new_matrix.shape[1]), key=lambda c: np.sum(np_counts[:, c]))
        ]
        ax = _alluvial.plot(
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


def agreement_score_bar_plot(
    adata,
    popv_prediction_key: Optional[str] = "popv_prediction",
    consensus_percentage_key: Optional[str] = "consensus_percentage",
    save_folder: Optional[str] = None,
):
    """
    Create bar-plot of prediction scores in query cells after running popv.

    Parameters
    ----------
    adata
        AnnData object.
    popv_prediction_score
        Key in adata.obs for prediction scores.
    save_folder
        Path to a folder for storing the plot. Defaults to None and plot is not stored.

    Returns
    -------
    Returns axis of corresponding plot.

    """
    celltypes = adata.obs[popv_prediction_key].unique()
    mean_agreement = [
        np.mean(
            adata[
                adata.obs["_dataset"] == "query" & adata.obs[popv_prediction_key] == x
            ]
            .obs[consensus_percentage_key]
            .astype(float)
        )
        for x in celltypes
    ]
    mean_agreement = pd.DataFrame(
        [mean_agreement], index=["agreement"], columns=celltypes
    ).T

    mean_agreement = mean_agreement.sort_values("agreement", ascending=True)
    ax = mean_agreement.plot.bar(y="agreement", figsize=(15, 2), legend=False)
    plt.ylabel("Mean Agreement")
    plt.xticks(rotation=290, ha="left")
    if save_folder is not None:
        figpath = os.path.join(save_folder, "percelltype_agreement_barplot.pdf")
        plt.savefig(figpath, bbox_inches="tight")
    return ax


def prediction_score_bar_plot(
    adata,
    popv_prediction_score: Optional[str] = "popv_prediction_score",
    save_folder: Optional[str] = None,
):
    """
    Create bar-plot of prediction scores in query cells after running popv.

    Parameters
    ----------
    adata
        AnnData object.
    popv_prediction_score
        Key in adata.obs for prediction scores.
    save_folder
        Path to a folder for storing the plot. Defaults to None and plot is not stored.

    Returns
    ----------
    Returns axis object of corresponding plot.

    """
    ax = (
        adata[adata.obs["_dataset"] == "query"]
        .obs[popv_prediction_score]
        .value_counts()
        .sort_index()
        .plot.bar()
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title("PopV Prediction Score")
    if save_folder is not None:
        save_path = os.path.join(save_folder, "prediction_score_barplot.pdf")
        ax.get_figure().savefig(save_path, bbox_inches="tight", dpi=300)
    return ax


def make_agreement_plots(
    adata,
    methods: list,
    popv_prediction_key: Optional[str] = "popv_prediction",
    save_folder: Optional[str] = None,
):
    """
    Create plot of confusion matrix for different popv methods and consensus prediction.

    Parameters
    ----------
    adata
        AnnData object.
    methods
        List with key for methods for which confusion matrix is computed.
    popv_prediction_key
        Key in adata.obs for consensus prediction.
    save_folder
        Path to a folder for storing the plot. Defaults to None and plot is not stored.

    Returns
    -------
    Returns axis of corresponding plot.

    """
    # clear all existing figures first
    # or else this will interfere with the pdf saving capabilities
    fig_nums = plt.get_fignums()
    for num in fig_nums:
        plt.close(num)

    for method in methods:
        logging.info(f"Making confusion matrix for {method}")
        _prediction_eval(
            adata.obs[method],
            adata.obs[popv_prediction_key],
            name=method,
            x_label=method,
            y_label=popv_prediction_key,
            res_dir=None,
        )


def _prediction_eval(
    pred,
    labels,
    name,
    x_label="",
    y_label="",
    res_dir="./",
):
    """Generate confusion matrix."""
    x = np.concatenate([labels, pred])
    types, _ = np.unique(x, return_inverse=True)
    prop = np.asarray([np.mean(np.asarray(labels) == i) for i in types])
    prop = pd.DataFrame([types, prop], index=["types", "prop"], columns=types).T
    mtx = confusion_matrix(labels, pred, normalize="true")
    df = pd.DataFrame(mtx, columns=types, index=types)
    df = df.loc[np.unique(labels), np.unique(pred)]
    df = df.rename_axis(x_label, axis="columns")
    df = df.rename_axis(y_label)
    plt.figure(figsize=(15, 12))
    sns.heatmap(df, linewidths=0.005, cmap="OrRd")
    plt.tight_layout()
    plt.title(name)
    if res_dir is not None:
        output_pdf_fn = os.path.join(res_dir, "confusion_matrices.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf_fn)
        for fig in range(1, plt.gcf().number + 1):
            pdf.savefig(fig)
        pdf.close()
    plt.show()

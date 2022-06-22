import functools
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

from .accuracy import *
from .methods import *
from .utils import *
from .utils import _check_nonnegative_integers
from .visualization import sample_report


def process_query(
    query_adata,
    ref_adata,
    save_folder,
    ref_adata_path,
    ref_labels_key,
    ref_batch_key="donor_method",
    ref_cell_ontology_key="final_annotation_cell_ontology_id",
    query_labels_key=None,
    query_batch_key=None,
    pretrained_scvi_path=None,
    unknown_celltype_label="unknown",
    training_mode="online",
    hvg=True,
    n_samples_per_label=100,
):
    """
    Processes the query dataset in preperation for the annotation pipeline.


    Parameters
    ----------
    query_adata
        AnnData of query cells
    save_folder
        Folder to save data to
    ref_adata_path
        Path to reference AnnData
    ref_labels_key
        Key in obs field of reference AnnData for labels
    ref_batch_keys
        List of Keys (or None) in obs field of reference AnnData to
        use for labels
    ref_layers_key:
        If not None, will use data from ref_adata.layers[ref_layers_key]
    ref_cell_ontology_key
        Key in obs field of reference AnnData for ontology ids
    query_batch_key
        Key in obs field of query adata for batch information.
    query_layers_key
        If not None, will use data from query_adata.layers[query_layers_key].
    query_labels_key
        Key in obs field of query adata for label information.
        This is only used for training scANVI.
        Make sure to set unknown_celltype_label to mark unlabelled cells.
    unknown_celltype_label
        If query_labels_key is not None, cells with label unknown_celltype_label
        will be treated as unknown and will be predicted by the model.
    pretrained_scvi_path
        Path to pretrained scvi model
    training_mode
        If training_mode=='offline', will train scVI and scANVI from scratch. Else if
        training_mode=='online' and all the genes in the pretrained models are present
        in query adata, will train the scARCHES version of scVI and scANVI, resulting in
        faster training times.
    hvg
        If True, subsets data to 4000 highly variable genes according to `sc.pp.highly_variable_genes`

    Returns
    -------
    adata
        AnnData object that is setup for use with the annotation pipeline

    """
    # TODO add check that varnames are all unique
    assert _check_nonnegative_integers(query_adata.X) == True
    assert _check_nonnegative_integers(ref_adata.X) == True

    if query_adata.n_obs == 0:
        raise ValueError("Input query anndata has no cells.")

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if isinstance(ref_batch_key, list):
        make_batch_covariate(
            ref_adata, ref_batch_key, new_batch_key="_batch_annotation"
        )
    else:
        ref_adata.obs["_batch_annotation"] = ref_adata.obs[ref_batch_key]

    ref_adata.obs["_labels_annotation"] = ref_adata.obs[ref_labels_key]
    ref_adata.obs["_dataset"] = "ref"
    ref_adata.layers["scvi_counts"] = ref_adata.X

    # subsample the reference cells used for training certain models
    ref_adata.obs["_ref_subsample"] = False
    ref_subsample_idx = subsample_dataset(
        ref_adata,
        ref_labels_key,
        n_samples_per_label=n_samples_per_label,
        ignore_label=[unknown_celltype_label],
    )
    ref_adata.obs.loc[ref_subsample_idx, "_ref_subsample"] = True

    if isinstance(query_batch_key, list):
        make_batch_covariate(
            query_adata, query_batch_key, new_batch_key="_batch_annotation"
        )
        query_batches = query_adata.obs["_batch_annotation"].astype("str")
        query_adata.obs["_batch_annotation"] = query_batches + "_query"
    elif query_batch_key is not None:
        query_batches = query_adata.obs[query_batch_key].astype("str")
        query_adata.obs["_batch_annotation"] = query_batches + "_query"
    else:
        query_adata.obs["_batch_annotation"] = "query"

    query_adata.obs["_dataset"] = "query"
    query_adata.obs["_ref_subsample"] = False
    query_adata.obs[ref_cell_ontology_key] = unknown_celltype_label
    query_adata.layers["scvi_counts"] = query_adata.X

    if query_labels_key is not None:
        query_labels = query_adata.obs[query_labels_key]
        known_cell_idx = np.where(query_labels != unknown_celltype_label)[0]
        if len(known_cell_idx) != 0:
            query_adata.obs["_labels_annotation"][known_cell_idx] = query_labels[
                known_cell_idx
            ]
        else:
            query_adata.obs["_labels_annotation"] = unknown_celltype_label
    else:
        query_adata.obs["_labels_annotation"] = unknown_celltype_label

    if training_mode == "online":
        query_adata = query_adata[:, ref_adata.var_names].copy()
        adata = anndata.concat((ref_adata, query_adata))
    elif training_mode == "offline":
        adata = anndata.concat((ref_adata, query_adata))

    adata.uns["_training_mode"] = training_mode

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['logcounts'] = adata.X.copy()
    sc.pp.scale(adata, max_value=10, zero_center=False)
    n_top_genes = np.min((4000, query_adata.n_vars))
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=True,
        layer="scvi_counts",
        flavor="seurat_v3",
    )
    sc.tl.pca(adata)
    # CSV's obs names x prediction by method, scvi latent space, scanvi latent space
    # Remove any 0 expression cells
    idx = [i[0] for i in np.argwhere(np.sum(adata.X.todense(), 1) == 0)]
    zero_cell_names = adata[idx].obs.index
    sc.pp.filter_cells(adata, min_counts=1, inplace=True)

    logging.warning(
        f"The following cells will be excluded from annotation because they have no expression:{zero_cell_names}, likely due to highly variable gene selection. We recommend you subset the data yourself and set hvg to False."
    )

    ref_query_results_fn = os.path.join(save_folder, "annotated_query_plus_ref.h5ad")
    # anndata.concat((query_adata, ref_adata), join="outer").write(ref_query_results_fn)
    adata.write(ref_query_results_fn)

    query_adata = adata[adata.obs._dataset == "query"]
    query_results_fn = os.path.join(save_folder, "annotated_query.h5ad")
    query_adata.write(query_results_fn)
    return adata


def prediction_eval(
    pred,
    labels,
    name,
    x_label="",
    y_label="",
    res_dir="./",
):
    """
    Generate confusion matrix
    """
    x = np.concatenate([labels, pred])
    types, temp = np.unique(x, return_inverse=True)
    prop = np.asarray([np.mean(np.asarray(labels) == i) for i in types])
    prop = pd.DataFrame([types, prop], index=["types", "prop"], columns=types).T
    mtx = confusion_matrix(labels, pred, normalize="true")
    df = pd.DataFrame(mtx, columns=types, index=types)
    df = df.loc[np.unique(labels), np.unique(pred)]
    df = df.rename_axis(
        x_label, axis="columns"
    )  # TODO: double check the axes are correct
    df = df.rename_axis(y_label)
    df.to_csv(res_dir + "/%s_prediction_accuracy.csv" % name)
    plt.figure(figsize=(15, 12))
    sns.heatmap(df, linewidths=0.005, cmap="OrRd")
    plt.tight_layout()
    output_pdf_fn = os.path.join(res_dir, "confusion_matrices.pdf")
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_pdf_fn)
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
    pdf.close()
    return plt.figure(1)


def compute_consensus(adata, prediction_keys):
    """
    Computes consensus prediction and statistics between all methods.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs for for predicted values

    Returns
    -------
    Saves the consensus prediction in adata.obs['consensus_prediction']
    Saves the consensus percentage between methods in adata.obs['consensus_percentage']
    """
    consensus_prediction = adata.obs[prediction_keys].apply(majority_vote, axis=1)
    adata.obs["consensus_prediction"] = consensus_prediction

    agreement = adata.obs[prediction_keys].apply(majority_count, axis=1)
    agreement *= 100
    adata.obs["consensus_percentage"] = agreement.values.round(2).astype(str)


def majority_vote(x):
    a, b = np.unique(x, return_counts=True)
    return a[np.argmax(b)]


def majority_count(x):
    a, b = np.unique(x, return_counts=True)
    return np.max(b)


def annotate_data(
    adata,
    methods,
    save_path,
    pretrained_scvi_path=None,
    pretrained_scanvi_path=None,
    onclass_ontology_file="cl.ontology",
    onclass_obo_fp="cl.obo",
    onclass_emb_fp="cl.ontology.nlp.emb",
    scvi_max_epochs=None,
    scanvi_max_epochs=None,
):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ref_query_results_fn = os.path.join(save_path, "annotated_query_plus_ref.h5ad")
    query_results_fn = os.path.join(save_path, "annotated_query.h5ad")
    if "bbknn" in methods:
        run_bbknn(adata, batch_key="_batch_annotation")
        run_knn_on_bbknn(
            adata, labels_key="_labels_annotation", result_key="knn_on_bbknn_pred"
        )

        save_results(
            adata,
            ref_query_results_fn,
            obs_keys=["knn_on_bbknn_pred"],
            obsm_keys=["bbknn_umap"],
        )
        save_results(
            adata,
            query_results_fn,
            obs_keys=["knn_on_bbknn_pred"],
            obsm_keys=["bbknn_umap"],
        )

    if "scvi" in methods:
        training_mode = adata.uns["_training_mode"]
        scvi_obsm_latent_key = "X_scvi_" + training_mode

        run_scvi(
            adata,
            max_epochs=scvi_max_epochs,
            n_latent=50,
            dropout_rate=0.1,
            dispersion="gene-batch",
            obsm_latent_key=scvi_obsm_latent_key,
            pretrained_scvi_path=pretrained_scvi_path,
        )
        knn_pred_key = "knn_on_scvi_{}_pred".format(training_mode)
        run_knn_on_scvi(adata, obsm_key=scvi_obsm_latent_key, result_key=knn_pred_key)
        save_results(
            adata,
            ref_query_results_fn,
            obs_keys=[knn_pred_key],
            obsm_keys=[scvi_obsm_latent_key, scvi_obsm_latent_key + "_umap"],
        )
        save_results(
            adata,
            query_results_fn,
            obs_keys=[knn_pred_key],
            obsm_keys=[scvi_obsm_latent_key, scvi_obsm_latent_key + "_umap"],
        )
        np.savetxt(
            os.path.join(save_path, "scvi_latent.csv"),
            adata.obsm[scvi_obsm_latent_key],
            delimiter=",",
        )

    if "scanvi" in methods:
        training_mode = adata.uns["_training_mode"]
        obsm_latent_key = "X_scanvi_{}".format(training_mode)
        predictions_key = "scanvi_{}_pred".format(training_mode)
        run_scanvi(
            adata,
            max_epochs=scanvi_max_epochs,
            n_latent=50,
            dropout_rate=0.1,
            dispersion='gene-batch',
            obsm_latent_key=obsm_latent_key,
            obs_pred_key=predictions_key,
            pretrained_scanvi_path=pretrained_scanvi_path,
        )

        save_results(
            adata,
            ref_query_results_fn,
            obs_keys=[predictions_key],
            obsm_keys=[obsm_latent_key],
        )
        save_results(
            adata,
            query_results_fn,
            obs_keys=[predictions_key],
            obsm_keys=[obsm_latent_key],
        )
        np.savetxt(
            os.path.join(save_path, "scanvi_latent.csv"),
            adata.obsm[obsm_latent_key],
            delimiter=",",
        )

    if "svm" in methods:
        run_svm_on_hvg(adata)
        save_results(adata, ref_query_results_fn, obs_keys=["svm_pred"])
        save_results(adata, query_results_fn, obs_keys=["svm_pred"])

    if "rf" in methods:
        run_rf_on_hvg(adata)
        save_results(adata, ref_query_results_fn, obs_keys=["rf_pred"])
        save_results(adata, query_results_fn, obs_keys=["rf_pred"])

    if "onclass" in methods:
        run_onclass(
            adata=adata,
            layer="logcounts",
            max_iter=20,
            cl_obo_file=onclass_obo_fp,
            cl_ontology_file=onclass_ontology_file,
            nlp_emb_file=onclass_emb_fp,
        )
        save_results(adata, ref_query_results_fn, obs_keys=["onclass_pred"])
        save_results(adata, query_results_fn, obs_keys=["onclass_pred"])

    if "scanorama" in methods:
        run_scanorama(adata, batch_key="_batch_annotation")
        run_knn_on_scanorama(adata)
        save_results(
            adata,
            ref_query_results_fn,
            obs_keys=["knn_on_scanorama_pred"],
            obsm_keys=["scanorama_umap"],
        )
        save_results(
            adata,
            query_results_fn,
            obs_keys=["knn_on_scanorama_pred"],
            obsm_keys=["scanorama_umap"],
        )

    # Here we compute the consensus statistics
    all_prediction_keys = [
        "knn_on_bbknn_pred",
        "knn_on_scvi_online_pred",
        "knn_on_scvi_offline_pred",
        "scanvi_online_pred",
        "scanvi_offline_pred",
        "svm_pred",
        "rf_pred",
        "onclass_pred",
        "knn_on_scanorama_pred",
    ]

    obs_keys = adata.obs.keys()
    pred_keys = [
        key for key in obs_keys if key in all_prediction_keys
    ]  # should this be all_prediction_keys or methods?

    compute_consensus(adata, pred_keys)
    ontology_vote_onclass(adata, onclass_obo_fp, "ontology_prediction", pred_keys)

    save_results(
        adata,
        ref_query_results_fn,
        obs_keys=["consensus_prediction", "consensus_percentage"],
    )

    save_results(
        adata,
        query_results_fn,
        obs_keys=["consensus_prediction", "consensus_percentage"],
    )
    print("Final annotated query plus ref saved at ", ref_query_results_fn)
    print("Final annotated query saved at ", query_results_fn)

    # CSV's obs names x prediction by method, scvi latent space, scanvi latent space
    adata[adata.obs._dataset == "query"].obs[
        pred_keys
        + ["consensus_prediction", "consensus_percentage", "ontology_prediction"]
    ].to_csv(os.path.join(save_path, "predictions.csv"))


def ontology_vote_onclass(adata, obofile, save_key, pred_keys):
    """
    Compute prediction using ontology aggregation method.
    """
    G = make_ontology_dag(obofile)
    cell_type_root_to_node = {}
    aggregate_ontology_pred = []
    depths = {"cell": 0}
    scores = []
    for cell in adata.obs.index:
        score = defaultdict(lambda: 0)
        score["cell"] = 0
        for pred_key in pred_keys:
            cell_type = adata.obs[pred_key][cell]
            if not pd.isna(cell_type):
                cell_type = cell_type.lower()
                if cell_type in cell_type_root_to_node:
                    root_to_node = cell_type_root_to_node[cell_type]
                else:
                    root_to_node = nx.descendants(G, cell_type)
                    cell_type_root_to_node[cell_type] = root_to_node
                if pred_key == "onclass_pred":
                    for ancestor_cell_type in root_to_node:
                        score[ancestor_cell_type] += 1
                        depths[ancestor_cell_type] = len(
                            nx.shortest_path(G, ancestor_cell_type, "cell")
                        )
                depths[cell_type] = len(nx.shortest_path(G, cell_type, "cell"))
                score[cell_type] += 1
        aggregate_ontology_pred.append(
            max(
                score,
                key=lambda k: (
                    score[k],
                    depths[k],
                    26 - string.ascii_lowercase.index(cell_type[0]),
                ),
            )
        )
        scores.append(score[cell_type])
    adata.obs[save_key] = aggregate_ontology_pred
    adata.obs[save_key + "_score"] = scores
    return adata


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


@deprecated
def ontology_vote_onclass_old(adata, dag, pred_keys, save_key):
    depths = calculate_depths(dag)
    save_key_score = save_key + "_score"

    adata.obs[save_key] = "na"
    adata.obs[save_key_score] = "na"

    for i, cell_name in enumerate(adata.obs_names):
        if i % 5000 == 0:
            print(i)

        cell = adata[cell_name]

        # make scores
        nx.set_node_attributes(dag, 0, "score")

        for k in pred_keys:
            celltype = cell.obs[k][0]

            if k == "onclass_pred":
                for node in nx.descendants(dag, celltype):
                    dag.nodes[node]["score"] += 1

            dag.nodes[celltype]["score"] += 1

        max_node = None
        max_score = 0
        max_depth = 1e8

        for node in dag.nodes(data="score"):
            score = node[1]
            celltype = node[0]
            if score != 0:
                if score > max_score:
                    max_node = node
                    max_score = score
                    max_depth = depths[celltype]

                if score == max_score:
                    depth = depths[celltype]
                    if depth > max_depth:
                        max_node = node
                        max_depth = depth
                        max_score = score

        cell_name = cell.obs_names[0]

        adata.obs[save_key][cell_name] = max_node[0]
        adata.obs[save_key_score][cell_name] = max_node[1]

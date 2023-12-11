"""Helper function to execute cell-type annotation and accumulate results."""

import inspect
import logging
import os
import pickle
import string
from collections import defaultdict

import anndata
import networkx as nx
import numpy as np
import pandas as pd

from popv import _utils, algorithms


def annotate_data(
    adata: anndata.AnnData,
    methods: list | None = None,
    save_path: str | None = None,
    methods_kwargs: dict | None = None,
) -> None:
    """
    Annotate an AnnData dataset preprocessed by preprocessing.Process_Query by using the annotation pipeline.

    Parameters
    ----------
    adata
        AnnData of query and reference cells. Adata object of Process_Query instance.

    Methods
    -------
        List of methods used for cell-type annotation. Defaults to all algorithms.
    save_path
        Path were annotated query data is saved. Defaults to None and is not saving data.
    methods_kwargs
        Dictionary, where keys are used methods and values contain non-default parameters.
        Default to empty-dictionary.
    """
    if save_path is not None and not os.path.exists(save_path):
        os.mkdir(save_path)
    methods = (
        methods
        if methods is not None
        else [i[0] for i in inspect.getmembers(algorithms, inspect.isclass)]
        if not adata.uns["_prediction_mode"] == "fast"
        else ["knn_on_scvi", "scanvi", "svm", "rf", "onclass", "celltypist"]
    )

    if adata.uns["_cl_obo_file"] is False and "onclass" in methods:
        methods.remove("onclass")

    methods_kwargs = methods_kwargs if methods_kwargs else {}

    all_prediction_keys = []
    all_prediction_keys_seen = []
    for method in methods:
        current_method = getattr(algorithms, method)(**methods_kwargs.pop(method, {}))
        current_method.compute_integration(adata)
        current_method.predict(adata)
        current_method.compute_embedding(adata)
        all_prediction_keys += [current_method.result_key]
        if hasattr(current_method, "seen_result_key"):
            all_prediction_keys_seen += [current_method.seen_result_key]
        else:
            all_prediction_keys_seen += [current_method.result_key]

    # Here we compute the consensus statistics
    logging.info(f"Using predictions {all_prediction_keys} for PopV consensus")
    adata.uns["prediction_keys"] = all_prediction_keys
    adata.uns["prediction_keys_seen"] = all_prediction_keys_seen
    compute_consensus(adata, all_prediction_keys_seen)
    # No ontology prediction if ontology is set to False.
    if adata.uns["_cl_obo_file"] is False:
        adata.obs[["popv_prediction", "popv_prediction_score"]] = adata.obs[
            ["popv_majority_vote_prediction", "popv_majority_vote_score"]
        ]
        adata.obs[["popv_prediction_parent"]] = adata.obs[["popv_majority_vote_prediction"]]
    else:
        ontology_vote_onclass(adata, all_prediction_keys)
        ontology_parent_onclass(adata, all_prediction_keys)

    if save_path is not None:
        prediction_save_path = os.path.join(save_path, "predictions.csv")
        adata[adata.obs._dataset == "query"].obs[
            all_prediction_keys
            + [
                "popv_prediction",
                "popv_prediction_score",
                "popv_majority_vote_prediction",
                "popv_majority_vote_score",
                "popv_parent",
            ]
        ].to_csv(prediction_save_path)

        logging.info(f"Predictions saved to {prediction_save_path}")


def compute_consensus(adata: anndata.AnnData, prediction_keys: list) -> None:
    """
    Compute consensus prediction and statistics between all methods.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs containing predicted cell_types.

    Returns
    -------
    Saves the consensus prediction in adata.obs['popv_majority_vote_prediction']
    Saves the consensus percentage between methods in adata.obs['popv_majority_vote_score']

    """
    consensus_prediction = adata.obs[prediction_keys].apply(_utils.majority_vote, axis=1)
    adata.obs["popv_majority_vote_prediction"] = consensus_prediction

    agreement = adata.obs[prediction_keys].apply(_utils.majority_count, axis=1)
    adata.obs["popv_majority_vote_score"] = agreement.values


def ontology_vote_onclass(
    adata: anndata.AnnData,
    prediction_keys: list,
    save_key: str | None = "popv_prediction",
):
    """
    Compute prediction using ontology aggregation method.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs containing predicted cell_types.
    save_key
        Name of the field in adata.obs to store the consensus prediction.

    Returns
    -------
    Saves the consensus prediction in adata.obs[save_key]
    Saves the consensus percentage between methods in adata.obs[save_key + '_score']
    Saves the overlap in original prediction in
    """
    if adata.uns["_prediction_mode"] == "retrain":
        G = _utils.make_ontology_dag(adata.uns["_cl_obo_file"])
        if adata.uns["_save_path_trained_models"] is not None:
            pickle.dump(G, open(adata.uns["_save_path_trained_models"] + "obo_dag.pkl", "wb"))
    else:
        G = pickle.load(open(adata.uns["_save_path_trained_models"] + "obo_dag.pkl", "rb"))

    cell_type_root_to_node = {}
    aggregate_ontology_pred = [None] * adata.n_obs
    depth = {"cell": 0}
    scores = [None] * adata.n_obs
    depths = [None] * adata.n_obs
    onclass_depth = [None] * adata.n_obs
    depth["cell"] = 0

    for ind, cell in enumerate(adata.obs.index):
        score = defaultdict(lambda: 0)
        score["cell"] = 0
        for pred_key in prediction_keys:
            cell_type = adata.obs[pred_key][cell]
            if not pd.isna(cell_type):
                if cell_type in cell_type_root_to_node:
                    root_to_node = cell_type_root_to_node[cell_type]
                else:
                    root_to_node = nx.descendants(G, cell_type)
                    cell_type_root_to_node[cell_type] = root_to_node
                    depth[cell_type] = len(nx.shortest_path(G, cell_type, "cell"))
                    for ancestor_cell_type in root_to_node:
                        depth[ancestor_cell_type] = len(nx.shortest_path(G, ancestor_cell_type, "cell"))
                if pred_key == "popv_onclass_prediction":
                    onclass_depth[ind] = depth[cell_type]
                    for ancestor_cell_type in root_to_node:
                        score[ancestor_cell_type] += 1
                score[cell_type] += 1
        # Find cell-type most present across all classifiers.
        # If tie then deepest in network.
        # If tie then last in alphabet, just to make it consistent across multiple cells.
        celltype_consensus = max(
            score,
            key=lambda k: (
                score[k],
                depth[k],
                26 - string.ascii_lowercase.index(cell_type[0].lower()),
            ),
        )
        aggregate_ontology_pred[ind] = celltype_consensus
        scores[ind] = score[celltype_consensus]
        depths[ind] = depth[celltype_consensus]
    adata.obs[save_key] = aggregate_ontology_pred
    adata.obs[save_key + "_score"] = scores
    adata.obs[save_key + "_depth"] = depths
    adata.obs[save_key + "_onclass_relative_depth"] = np.array(onclass_depth) - adata.obs[save_key + "_depth"]
    # Change numeric values to categoricals.
    adata.obs[[save_key + "_score", save_key + "_depth", save_key + "_onclass_relative_depth"]] = adata.obs[
        [save_key + "_score", save_key + "_depth", save_key + "_onclass_relative_depth"]
    ].astype("category")
    return adata


def ontology_parent_onclass(
    adata: anndata.AnnData,
    prediction_keys: list,
    save_key: str = "popv_parent",
    allowed_errors: int = 2,
):
    """
    Compute common parent consensus prediction using ontology accumulation.

    Parameters
    ----------
    adata
        AnnData object
    prediction_keys
        Keys in adata.obs containing predicted cell_types.
    save_key
        Name of the field in adata.obs to store the consensus prediction. Default to 'popv_parent'.
    allowed_errors
        How many misclassifications are allowed to find common ontology ancestor. Defaults to 2.

    Returns
    -------
    Saves the consensus prediction in adata.obs[save_key]
    Saves the consensus percentage between methods in adata.obs[save_key + '_score']
    Saves the overlap in original prediction in
    """
    if adata.uns["_prediction_mode"] == "retrain":
        G = _utils.make_ontology_dag(adata.uns["_cl_obo_file"])
        if adata.uns["_save_path_trained_models"] is not None:
            pickle.dump(G, open(adata.uns["_save_path_trained_models"] + "obo_dag.pkl", "wb"))
    else:
        G = pickle.load(open(adata.uns["_save_path_trained_models"] + "obo_dag.pkl", "rb"))

    cell_type_root_to_node = {}
    aggregate_ontology_pred = []
    depth = {"cell": 0}
    for cell in adata.obs.index:
        score = defaultdict(lambda: 0)
        score_popv = defaultdict(lambda: 0)
        score["cell"] = 0
        for pred_key in prediction_keys:
            cell_type = adata.obs[pred_key][cell]
            if not pd.isna(cell_type):
                if cell_type in cell_type_root_to_node:
                    root_to_node = cell_type_root_to_node[cell_type]
                else:
                    root_to_node = nx.descendants(G, cell_type)
                    cell_type_root_to_node[cell_type] = root_to_node
                    depth[cell_type] = len(nx.shortest_path(G, cell_type, "cell"))
                    for ancestor_cell_type in root_to_node:
                        depth[ancestor_cell_type] = len(nx.shortest_path(G, ancestor_cell_type, "cell"))
                for ancestor_cell_type in list(root_to_node) + [cell_type]:
                    score[ancestor_cell_type] += 1
                score_popv[cell_type] += 1
        score = {key: min(len(prediction_keys) - allowed_errors, value) for key, value in score.items()}

        # Find ancestor most present and deepest across all classifiers.
        # If tie, then highest in original classifier.
        # If tie then last in alphabet, just to make it consistent across multiple cells.
        celltype_consensus = max(
            score,
            key=lambda k: (
                score[k],
                depth[k],
                score_popv[k],
                26 - string.ascii_lowercase.index(cell_type[0].lower()),
            ),
        )
        aggregate_ontology_pred.append(celltype_consensus)
    adata.obs[save_key] = aggregate_ontology_pred
    return adata

"""Helper function to execute cell-type annotation and accumulate results."""

import inspect
import logging
import os
import string
from collections import defaultdict
from typing import Optional

import anndata
import networkx as nx
import pandas as pd

from popv import _utils, algorithms


def annotate_data(
    adata: anndata.AnnData,
    methods: Optional[list] = None,
    save_path: Optional[str] = None,
    methods_kwargs: Optional[dict] = None,
) -> None:
    """
    Annotate an AnnData dataset preprocessed by preprocessing.Process_Query by using the annotation pipeline.

    Parameters
    ----------
    adata
        AnnData of query and reference cells. Adata object of Process_Query instance.
    methods
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
        if methods
        else [i[0] for i in inspect.getmembers(algorithms, inspect.isclass)]
    )
    methods_kwargs = methods_kwargs if methods_kwargs else {}

    all_prediction_keys = []
    for method in methods:
        current_method = getattr(algorithms, method)(**methods_kwargs.pop(method, {}))
        current_method.compute_integration(adata)
        current_method.predict(adata)
        current_method.compute_embedding(adata)
        all_prediction_keys += [current_method.result_key]

    # Here we compute the consensus statistics
    logging.info(f"Using predictions {all_prediction_keys} for PopV consensus")
    adata.uns["prediction_keys"] = all_prediction_keys
    compute_consensus(adata, all_prediction_keys)
    ontology_vote_onclass(adata, all_prediction_keys)

    if save_path is not None:
        prediction_save_path = os.path.join(save_path, "predictions.csv")
        adata[adata.obs._dataset == "query"].obs[
            all_prediction_keys
            + [
                "popv_prediction",
                "popv_prediction_score",
                "popv_majority_vote_prediction",
                "popv_majority_vote_score",
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
    ----------
    Saves the consensus prediction in adata.obs['popv_majority_vote_prediction']
    Saves the consensus percentage between methods in adata.obs['popv_majority_vote_score']

    """
    consensus_prediction = adata.obs[prediction_keys].apply(
        _utils.majority_vote, axis=1
    )
    adata.obs["popv_majority_vote_prediction"] = consensus_prediction

    agreement = adata.obs[prediction_keys].apply(_utils.majority_count, axis=1)
    adata.obs["popv_majority_vote_score"] = agreement.values.round(2).astype(str)


def ontology_vote_onclass(
    adata: anndata.AnnData,
    prediction_keys: list,
    save_key: Optional[str] = "popv_prediction",
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
    ----------
    Saves the consensus prediction in adata.obs[save_key]
    Saves the consensus percentage between methods in adata.obs[save_key + '_score']
    """
    G = _utils.make_ontology_dag(adata.uns["_cl_obo_file"])
    cell_type_root_to_node = {}
    aggregate_ontology_pred = []
    depths = {"cell": 0}
    scores = []
    for cell in adata.obs.index:
        score = defaultdict(lambda: 0)
        score["cell"] = 0
        for pred_key in prediction_keys:
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

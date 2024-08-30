from __future__ import annotations

import logging

import celltypist
import scanpy as sc

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class CELLTYPIST(BaseAlgorithm):
    """
    Class to compute Celltypist classifier.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
    labels_key
        Key in obs field of adata for cell-type information.
    result_key
        Key in obs in which celltype annotation results are stored.
    method_dict
        Additional parameters for celltypist training. Options at celltypist.train
    classifier_dict
        Dictionary to supply non-default values for celltypist annotation. Options at celltypist.annotate
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_celltypist_prediction",
        method_dict: dict | None = None,
        classifier_dict: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
        )

        if classifier_dict is None:
            classifier_dict = {}
        if method_dict is None:
            method_dict = {}

        self.method_dict = {"check_expression": False, "n_jobs": 10, "max_iter": 500}
        if method_dict is not None:
            self.method_dict.update(method_dict)

        self.classifier_dict = {"mode": "best match", "majority_voting": True}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def _predict(self, adata):
        logging.info(f'Saving celltypist results to adata.obs["{self.result_key}"]')

        flavor = 'rapids' if settings.cuml else 'vtraag'
        method = 'rapids' if settings.cuml else 'umap'
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca', method=method)
        sc.tl.louvain(adata, resolution=25., key_added='over_clustering', flavor=flavor)

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            print(len(train_idx))
            if len(train_idx) > 100000 and not True: # settings.cuml:
                self.method_dict['use_SGD'] = True
                self.method_dict['mini_batch'] = True

            train_adata = adata[train_idx].copy()
            model = celltypist.train(train_adata, self.labels_key, use_GPU=settings.cuml, **self.method_dict,)

            if adata.uns["_save_path_trained_models"]:
                model.write(adata.uns["_save_path_trained_models"] + "celltypist.pkl")
        if adata.uns["_prediction_mode"] == "fast":
            self.classifier_dict["majority_voting"] = False
        predictions = celltypist.annotate(
            adata,
            model=adata.uns["_save_path_trained_models"] + "celltypist.pkl",
            over_clustering=adata.obs['over_clustering'],
            **self.classifier_dict,
        )
        out_column = (
            "majority_voting" if "majority_voting" in predictions.predicted_labels.columns else "predicted_labels"
        )

        adata.obs[self.result_key] = predictions.predicted_labels[out_column]
        if self.return_probabilities:
            adata.obs[
                self.result_key + "_probabilities"
            ] = predictions.probability_matrix.max(axis=1).values

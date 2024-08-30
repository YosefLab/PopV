from __future__ import annotations

import logging

import anndata
import numpy as np
import scanorama
import scanpy as sc
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class SCANORAMA(BaseAlgorithm):
    """
    Class to compute KNN classifier after BBKNN integration.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
    labels_key
        Key in obs field of adata for cell-type information.
    result_key
        Key in obs in which celltype annotation results are stored.
    embedding_key
        Key in obsm in which UMAP embedding of integrated data is stored.
    method_dict
        Additional parameters for SCANORAMA. Options at scanorama.integrate_scanpy
    classifier_dict
        Dictionary to supply non-default values for KNN classifier. n_neighbors and weights supported.
    embedding_kwargs
        Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_on_scanorama_prediction",
        embedding_key: str | None = "X_umap_scanorma_popv",
        method_dict: dict | None = None,
        classifier_dict: dict | None = None,
        embedding_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
        )

        if embedding_kwargs is None:
            embedding_kwargs = {}
        if classifier_dict is None:
            classifier_dict = {}
        if method_dict is None:
            method_dict = {}

        self.method_dict = {"dimred": 50}
        if method_dict is not None:
            self.method_dict.update(method_dict)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

        self.embedding_kwargs = {
            "min_dist": 0.1,
        }
        self.embedding_kwargs.update(embedding_kwargs)

    def _compute_integration(self, adata):
        logging.info("Integrating data with scanorama")

        _adatas = [adata[adata.obs[self.batch_key] == i] for i in np.unique(adata.obs[self.batch_key])]
        scanorama.integrate_scanpy(_adatas, **self.method_dict)
        tmp_adata = anndata.concat(_adatas)
        adata.obsm["X_scanorama"] = tmp_adata[adata.obs_names].obsm["X_scanorama"]

    def _predict(self, adata, result_key="popv_knn_on_scanorama_prediction"):
        logging.info(f'Saving knn on scanorama results to adata.obs["{result_key}"]')

        ref_idx = adata.obs["_labelled_train_indices"]
        train_X = adata[ref_idx].obsm["X_scanorama"]
        train_Y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()

        if settings.cuml:
            from cuml.neighbors import KNeighborsClassifier as cuKNeighbors
            knn = cuKNeighbors(n_neighbors=self.classifier_dict["n_neighbors"])
        else:
            knn = make_pipeline(
                PyNNDescentTransformer(
                    n_neighbors=self.classifier_dict["n_neighbors"],
                    parallel_batch_queries=True,
                ),
                KNeighborsClassifier(
                    metric="precomputed", weights=self.classifier_dict["weights"]
                ),
            )

        knn.fit(train_X, train_Y)
        knn_pred = knn.predict(adata.obsm["X_scanorama"])

        # save_results
        adata.obs[self.result_key] = adata.obs[self.labels_key].cat.categories[knn_pred]

        if self.return_probabilities:
            adata.obs[self.result_key + "_probabilities"] = np.max(
                knn.predict_proba(adata.obsm["X_scanorama"]), axis=1
            )

    def _compute_embedding(self, adata):
        if self.compute_embedding:
            logging.info(
                f'Saving UMAP of scanorama results to adata.obs["{self.embedding_key}"]'
            )

            method = 'rapids' if settings.cuml else 'umap'
            sc.pp.neighbors(adata, use_rep="X_scanorama", method=method)
            adata.obsm[self.embedding_key] = sc.tl.umap(
                adata, copy=True, method=method, **self.embedding_kwargs
            ).obsm["X_umap"]

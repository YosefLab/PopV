import logging
from typing import Optional

import anndata
import numpy as np
import scanorama
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier


class SCANORAMA:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        result_key: Optional[str] = "popv_knn_on_scanorama_prediction",
        embedding_key: Optional[str] = "X_umap_scanorma_popv",
        method_dict: Optional[dict] = {},
        classifier_dict: Optional[dict] = {},
        embedding_dict: Optional[dict] = {},
    ) -> None:
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
            Dictionary to supply non-default values for KNN classifier. Options at sklearn.neighbors.KNeighborsClassifier
        embedding_dict
            Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
        """

        self.batch_key = batch_key
        self.labels_key = labels_key
        self.result_key = result_key
        self.embedding_key = embedding_key

        self.method_dict = {"dimred": 50}
        self.method_dict.update(method_dict)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        self.classifier_dict.update(classifier_dict)

        self.embedding_dict = {"min_dist": 0.1}
        self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with scanorama")

        _adatas = [
            adata[adata.obs[self.batch_key] == i]
            for i in np.unique(adata.obs[self.batch_key])
        ]
        scanorama.integrate_scanpy(_adatas, **self.method_dict)
        tmp_adata = anndata.concat(_adatas)
        adata.obsm["X_scanorama"] = tmp_adata[adata.obs_names].obsm["X_scanorama"]

    def predict(self, adata, result_key="popv_knn_on_scanorama_prediction"):
        logging.info(f'Saving knn on scanorama results to adata.obs["{result_key}"]')

        ref_idx = adata.obs["_dataset"] == "ref"
        query_idx = adata.obs["_dataset"] == "query"

        train_X = adata[ref_idx].obsm["X_scanorama"]
        train_Y = adata[ref_idx].obs[self.labels_key].to_numpy()
        test_X = adata[query_idx].obsm["X_scanorama"]

        knn = KNeighborsClassifier(**self.classifier_dict)
        knn.fit(train_X, train_Y)
        knn_pred = knn.predict(test_X)

        # save_results
        adata.obs[result_key] = adata.obs[self.labels_key]
        adata.obs.loc[query_idx, result_key] = knn_pred

    def compute_embedding(self, adata, embedding_key="X_scanorama_umap_popv"):
        logging.info(
            f'Saving UMAP of scanorama results to adata.obs["{embedding_key}"]'
        )

        sc.pp.neighbors(adata, use_rep="X_scanorama")
        adata.obsm[embedding_key] = sc.tl.umap(
            adata, copy=True, **self.embedding_dict
        ).obsm["X_umap"]

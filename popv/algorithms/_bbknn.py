import logging
from typing import Optional

import numpy as np
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier


class BBKNN:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        result_key: Optional[str] = "popv_knn_on_bbknn_prediction",
        embedding_key: Optional[str] = "X_umap_bbknn_popv",
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
            Additional parameters for BBKNN. Options at sc.external.pp.bbknn
        classifier_dict
            Dictionary to supply non-default values for KNN classifier. Options at sklearn.neighbors.KNeighborsClassifier
        embedding_dict
            Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
        """

        self.batch_key = batch_key
        self.labels_key = labels_key
        self.result_key = result_key
        self.embedding_key = embedding_key

        self.method_dict = {"metric": "angular", "n_pcs": 20}
        self.method_dict.update(method_dict)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        self.classifier_dict.update(classifier_dict)

        self.embedding_dict = {"min_dist": 0.01}
        self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with bbknn")

        sc.external.pp.bbknn(adata, batch_key=self.batch_key, **self.method_dict)

    def predict(self, adata):
        logging.info(f'Saving knn on bbknn results to adata.obs["{self.result_key}"]')

        distances = adata.obsp["distances"]

        ref_idx = adata.obs["_dataset"] == "ref"
        query_idx = adata.obs["_dataset"] == "query"

        ref_dist_idx = np.where(ref_idx)[0]
        query_dist_idx = np.where(query_idx)[0]

        train_y = adata.obs.loc[ref_idx, self.labels_key].to_numpy()
        train_distances = distances[ref_dist_idx, :][:, ref_dist_idx]

        knn = KNeighborsClassifier(metric="precomputed", **self.classifier_dict)
        knn.fit(train_distances, y=train_y)

        test_distances = distances[query_dist_idx, :][:, ref_dist_idx]
        knn_pred = knn.predict(test_distances)

        # save_results. ref cells get ref annotations, query cells get predicted
        adata.obs[self.result_key] = adata.obs[self.labels_key]
        adata.obs.loc[query_idx, self.result_key] = knn_pred

    def compute_embedding(self, adata):
        logging.info(
            f'Saving UMAP of bbknn results to adata.obs["{self.embedding_key}"]'
        )

        adata.obsm[self.embedding_key] = sc.tl.umap(
            adata, copy=True, **self.embedding_dict
        ).obsm["X_umap"]

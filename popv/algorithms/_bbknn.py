import logging

import numpy as np
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier


class BBKNN:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_knn_on_bbknn_prediction",
        embedding_key: str | None = "X_bbknn_umap_popv",
        method_dict: dict | None = None,
        classifier_dict: dict | None = None,
        embedding_dict: dict | None = None,
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

        self.method_dict = {
            "metric": "angular",
            "n_pcs": 50,
            "neighbors_within_batch": 8,
        }
        if method_dict is not None:
            self.method_dict.update(method_dict)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

        self.embedding_dict = {"min_dist": 0.1}
        if embedding_dict is not None:
            self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with bbknn")

        sc.external.pp.bbknn(adata, batch_key=self.batch_key, **self.method_dict)

    def predict(self, adata):
        logging.info(f'Saving knn on bbknn results to adata.obs["{self.result_key}"]')

        distances = adata.obsp["distances"]

        ref_idx = adata.obs["_dataset"] == "ref"

        ref_dist_idx = np.where(ref_idx)[0]

        train_y = adata.obs.loc[ref_idx, self.labels_key].to_numpy()

        train_distances = distances[ref_dist_idx, :][:, ref_dist_idx]
        test_distances = distances[:, :][:, ref_dist_idx]

        # Make sure BBKNN found the required number of neighbors, otherwise reduce n_neighbors for KNN.
        smallest_neighbor_graph = np.min(
            [
                np.diff(test_distances.indptr).min(),
                np.diff(train_distances.indptr).min(),
            ]
        )
        if smallest_neighbor_graph < 15:
            logging.warning(f"BBKNN found only {smallest_neighbor_graph} neighbors. Reduced neighbors in KNN.")
            self.classifier_dict["n_neighbors"] = smallest_neighbor_graph

        knn = KNeighborsClassifier(metric="precomputed", **self.classifier_dict)
        knn.fit(train_distances, y=train_y)

        adata.obs[self.result_key] = knn.predict(test_distances)

        if adata.uns["_return_probabilities"]:
            adata.obs[self.result_key + "_probabilities"] = np.max(knn.predict_proba(test_distances), axis=1)

    def compute_embedding(self, adata):
        if adata.uns["_compute_embedding"]:
            logging.info(f'Saving UMAP of bbknn results to adata.obs["{self.embedding_key}"]')
            adata.obsm[self.embedding_key] = sc.tl.umap(adata, copy=True, **self.embedding_dict).obsm["X_umap"]

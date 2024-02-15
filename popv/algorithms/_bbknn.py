import logging
from typing import Optional

import numpy as np
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier
from popv import settings


class BBKNN:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        result_key: Optional[str] = "popv_knn_on_bbknn_prediction",
        embedding_key: Optional[str] = "X_bbknn_umap_popv",
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
        self.enable_cuml = settings.cuml

        self.method_dict = {
            "metric": "euclidean" if self.enable_cuml else "cosine",
            "approx": not self.enable_cuml, # FAISS if cuml
            "n_pcs": 50,
            "neighbors_within_batch": 3 if self.enable_cuml else 8,
            "use_annoy": False, #pynndescent
        }
        self.method_dict.update(method_dict)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        self.classifier_dict.update(classifier_dict)

        self.embedding_dict = {
            "min_dist": 0.1}
        self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with bbknn")
        if len(adata.obs[self.batch_key].unique()) > 100 and self.enable_cuml:
            logging.warning('Using PyNNDescent instead of RAPIDS as high number of batches leads to OOM.')
            self.method_dict['approx'] = True
        print(self.method_dict)
        sc.external.pp.bbknn(adata, batch_key=self.batch_key, **self.method_dict)

    def predict(self, adata):
        logging.info(f'Saving knn on bbknn results to adata.obs["{self.result_key}"]')

        distances = adata.obsp["distances"]

        ref_idx = adata.obs["_labelled_train_indices"]

        ref_dist_idx = np.where(ref_idx)[0]

        train_y = adata.obs.loc[ref_idx, self.labels_key].cat.codes.to_numpy()

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
            logging.warning(
                f"BBKNN found only {smallest_neighbor_graph} neighbors. Reduced neighbors in KNN."
            )
            self.classifier_dict["n_neighbors"] = smallest_neighbor_graph

        knn = KNeighborsClassifier(metric="precomputed", **self.classifier_dict)
        knn.fit(train_distances, y=train_y)

        adata.obs[self.result_key] = adata.obs[self.labels_key].cat.categories[knn.predict(test_distances)]

        if adata.uns["_return_probabilities"]:
            adata.obs[self.result_key + "_probabilities"] = np.max(
                knn.predict_proba(test_distances), axis=1
            )

    def compute_embedding(self, adata):
        if adata.uns["_compute_embedding"]:
            logging.info(
                f'Saving UMAP of bbknn results to adata.obs["{self.embedding_key}"]'
            )
            if len(adata.obs[self.batch_key]) < 30 and settings.cuml:
                method = 'rapids'
            else:
                logging.warning('Using UMAP instead of RAPIDS as high number of batches leads to OOM.')
                method = 'umap'
                # RAPIDS not possible here as number of batches drastically increases GPU RAM.
            adata.obsm[self.embedding_key] = sc.tl.umap(
                adata, copy=True, method=method, **self.embedding_dict
            ).obsm["X_umap"]

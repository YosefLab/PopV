# Method that hightlights how to implement a new classifier. All class items are necessary but can contain Pass as only function argument.

import logging
from typing import Optional

# Import classifier here.


class NEW_ALGORITHM:
    # Remove embedding key if not an integration method.
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        layers_key: Optional[str] = None,
        result_key: Optional[str] = "popv_knn_on_scanorama_prediction",
        embedding_key: Optional[str] = "X_umap_scanorma_popv",
        method_dict: Optional[dict] = {},
        classifier_dict: Optional[dict] = {},
        embedding_dict: Optional[dict] = {},
    ) -> None:
        """
        Scaffold to set up new algorithm.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
        labels_key
            Key in obs field of adata for cell-type information.
        layers_key
            Layer in adata used for Onclass prediction.
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
        self.layers_key = layers_key

        self.result_key = result_key
        self.embedding_key = embedding_key

        # Necessary for integration method. Contains parameters for integration method.
        self.method_dict = {}
        self.method_dict.update(method_dict)

        # Necessary for classifier. Contains parameters for classifier.
        self.classifier_dict = {}
        self.classifier_dict.update(classifier_dict)

        # Necessary for integration method. Contains parameters for UMAP embedding.
        self.embedding_dict = {}
        self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with new integration method")

        # adata.obsm["X_new_method"] = embedded_data

    def predict(self, adata):
        logging.info(
            'Computing new classifier method. Storing prediction in adata.obs["{}"]'.format(
                self.result_key
            )
        )
        # adata.obs[self.result_key] = classifier_results

    def compute_embedding(self, adata):
        if adata.uns["_compute_embedding"]:
            logging.info(
                f'Saving UMAP of new integration method to adata.obs["{self.embedding_key}"]'
            )
            # sc.pp.neighbors(adata, use_rep="embedding_space")
            # adata.obsm[self.embedding_key] = sc.tl.umap(
            #     adata, copy=True, **self.embedding_dict
            # ).obsm["X_umap"]

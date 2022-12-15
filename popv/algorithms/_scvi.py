import logging
from typing import Optional

import numpy as np
import scanpy as sc
from scvi.model import SCVI
from sklearn.neighbors import KNeighborsClassifier


class SCVI_POPV:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        max_epochs: Optional[int] = None,
        use_gpu: Optional[bool] = False,
        save_folder: Optional[str] = None,
        result_key: Optional[str] = "popv_knn_on_scvi_prediction",
        embedding_key: Optional[str] = "X_scvi_umap_popv",
        model_kwargs: Optional[dict] = {},
        classifier_dict: Optional[dict] = {},
        embedding_dict: Optional[dict] = {},
    ) -> None:
        """
        Class to compute KNN classifier after scVI integration.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
        labels_key
            Key in obs field of adata for cell-type information.
        max_epochs
            Number of epochs scvi is trained.
        use_gpu
            Whether gpu is used for training.
        result_key
            Key in obs in which celltype annotation results are stored.
        embedding_key
            Key in obsm in which UMAP embedding of integrated data is stored.
        model_kwargs
            Dictionary to supply non-default values for SCVI model. Options at scvi.model.SCVI
        classifier_dict
            Dictionary to supply non-default values for KNN classifier. Options at sklearn.neighbors.KNeighborsClassifier
        embedding_dict
            Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.result_key = result_key
        self.embedding_key = embedding_key

        self.max_epochs = max_epochs
        self.use_gpu = use_gpu
        self.save_folder = save_folder

        self.model_kwargs = {
            "dropout_rate": 0.1,
            "dispersion": "gene",
            "n_layers": 2,
            "n_latent": 50,
        }
        self.model_kwargs.update(model_kwargs)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        self.classifier_dict.update(classifier_dict)

        self.embedding_dict = {"min_dist": 0.01}
        self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with scvi")

        SCVI.setup_anndata(
            adata,
            batch_key=self.batch_key,
            labels_key=self.labels_key,
            layer="scvi_counts",
            size_factor_key="size_factor"
        )
        pretrained_scvi_path = adata.uns["_pretrained_scvi_path"]

        if pretrained_scvi_path is None:
            model = SCVI(adata, **self.model_kwargs)
            logging.info("Training scvi offline.")
        else:
            query = adata[adata.obs["_dataset"] == "query"].copy()
            model = SCVI.load_query_data(query, pretrained_scvi_path)
            logging.info("Training scvi online.")

        if self.max_epochs is None:
            self.max_epochs = np.min([round((20000 / adata.n_obs) * 200), 200])

        model.train(
            max_epochs=self.max_epochs, train_size=0.9, use_gpu=adata.uns["_use_gpu"]
        )

        adata.obsm["X_scvi"] = model.get_latent_representation(adata)

        if self.save_folder is not None:
            model.save(self.save_folder, overwrite=True, save_anndata=False)

    def predict(self, adata):
        logging.info(f'Saving knn on scvi results to adata.obs["{self.result_key}"]')

        ref_idx = adata.obs["_dataset"] == "ref"
        query_idx = adata.obs["_dataset"] == "query"

        train_X = adata[ref_idx].obsm["X_scvi"]
        train_Y = adata[ref_idx].obs[self.labels_key].to_numpy()
        test_X = adata[query_idx].obsm["X_scvi"]

        knn = KNeighborsClassifier(**self.classifier_dict)
        knn.fit(train_X, train_Y)
        knn_pred = knn.predict(test_X)

        # save_results
        adata.obs[self.result_key] = adata.obs[self.labels_key]
        adata.obs.loc[query_idx, self.result_key] = knn_pred

    def compute_embedding(self, adata):
        logging.info(
            f'Saving UMAP of scvi results to adata.obs["{self.embedding_key}"]'
        )

        sc.pp.neighbors(adata, use_rep="X_scvi")
        adata.obsm[self.embedding_key] = sc.tl.umap(
            adata, copy=True, **self.embedding_dict
        ).obsm["X_umap"]

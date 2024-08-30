from __future__ import annotations

import logging
import pickle

import numpy as np
import scanpy as sc
from pynndescent import PyNNDescentTransformer
from scvi.model import SCVI
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class SCVI_POPV(BaseAlgorithm):
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        save_folder: str | None = None,
        result_key: str | None = "popv_knn_on_scvi_prediction",
        embedding_key: str | None = "X_scvi_umap_popv",
        model_kwargs: dict | None = None,
        classifier_dict: dict | None = None,
        embedding_dict: dict | None = None,
        train_kwargs: dict | None = None,
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
        result_key
            Key in obs in which celltype annotation results are stored.
        embedding_key
            Key in obsm in which UMAP embedding of integrated data is stored.
        model_kwargs
            Dictionary to supply non-default values for SCVI model. Options at scvi.model.SCVI
        classifier_dict
            Dictionary to supply non-default values for KNN classifier. n_neighbors and weights supported.
        embedding_dict
            Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
        train_kwargs
            Dictionary to supply non-default values for training scvi. Options at scvi.model.SCVI.train
        """
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            embedding_key=embedding_key,
        )
        if embedding_dict is None:
            embedding_dict = {}
        if classifier_dict is None:
            classifier_dict = {}
        if model_kwargs is None:
            model_kwargs = {}
        if train_kwargs is None:
            train_kwargs = {}
        self.save_folder = save_folder

        self.model_kwargs = {
            "dropout_rate": 0.05,
            "dispersion": "gene",
            "n_layers": 3,
            "n_latent": 20,
            "gene_likelihood": "nb",
            "use_batch_norm": "none",
            "use_layer_norm": "both",
            "encode_covariates": True,
        }

        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        self.classifier_dict = {"weights": "uniform", "n_neighbors": 15}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

        self.train_kwargs = {
            "max_epochs": 20,
            "batch_size": 512,
            "train_size": 1.0,
            "accelerator": settings.accelerator,
            "plan_kwargs" : {"n_epochs_kl_warmup": 20}
        }
        self.train_kwargs.update(train_kwargs)
        self.max_epochs = train_kwargs.get("max_epochs", None)

        self.embedding_dict = {"min_dist": 0.3}
        if embedding_dict is not None:
            self.embedding_dict.update(embedding_dict)

    def _compute_integration(self, adata):
        logging.info("Integrating data with scvi")

        # Go through obs field with subsampling information and subsample label information.
        if "subsampled_labels" not in adata.obs.columns:
            adata.obs["subsampled_labels"] = [
                label if subsampled else adata.uns["unknown_celltype_label"]
                for label, subsampled in zip(adata.obs["_labels_annotation"], adata.obs["_ref_subsample"])
            ]
        adata.obs["subsampled_labels"] = adata.obs["subsampled_labels"].astype("category")

        if not adata.uns["_pretrained_scvi_path"]:
            SCVI.setup_anndata(
                adata,
                batch_key=self.batch_key,
                labels_key="subsampled_labels",
                layer="scvi_counts",
            )
            model = SCVI(adata, **self.model_kwargs)
            logging.info("Training scvi offline.")
        else:
            query = adata[adata.obs["_dataset"] == "query"].copy()
            model = SCVI.load_query_data(query, adata.uns["_pretrained_scvi_path"])
            logging.info("Training scvi online.")

        if adata.uns["_prediction_mode"] == "fast":
            if self.max_epochs is None:
                self.train_kwargs['max_epochs'] = 1
            model.train(
                **self.train_kwargs
            )
        else:
            if self.max_epochs is None:
                self.max_epochs = min(round((20000 / adata.n_obs) * 200), 200)
            model.train(
                **self.train_kwargs
            )

            if (
                adata.uns["_save_path_trained_models"]
                and adata.uns["_prediction_mode"] == "retrain"
            ):
                # Update scvi for scanvi.
                adata.uns["_pretrained_scvi_path"] = adata.uns["_save_path_trained_models"] + "/scvi"
                model.save(
                    adata.uns["_save_path_trained_models"] + "/scvi",
                    save_anndata=False,
                    overwrite=True,
                )

        adata.obsm["X_scvi"] = model.get_latent_representation(adata)

    def _predict(self, adata):
        logging.info(f'Saving knn on scvi results to adata.obs["{self.result_key}"]')

        if adata.uns["_prediction_mode"] == "retrain":
            ref_idx = adata.obs["_labelled_train_indices"]

            train_X = adata[ref_idx].obsm["X_scvi"]
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
            if adata.uns["_save_path_trained_models"]:
                pickle.dump(
                    knn,
                    open(
                        adata.uns["_save_path_trained_models"] + "scvi_knn_classifier.pkl",
                        "wb",
                    ),
                )
        else:
            knn = pickle.load(
                open(
                    adata.uns["_save_path_trained_models"] + "scvi_knn_classifier.pkl",
                    "rb",
                )
            )

        knn_pred = knn.predict(adata.obsm["X_scvi"])

        # save_results
        adata.obs[self.result_key] = adata.obs[self.labels_key].cat.categories[knn_pred]
        if self.return_probabilities:
            adata.obs[self.result_key + "_probabilities"] = np.max(
                knn.predict_proba(adata.obsm["X_scvi"]), axis=1
            )

    def _compute_embedding(self, adata):
        if self.compute_embedding:
            logging.info(
                f'Saving UMAP of scvi results to adata.obs["{self.embedding_key}"]'
            )
            method = 'rapids' if settings.cuml else 'umap'
            sc.pp.neighbors(adata, use_rep="X_scvi", method=method)
            adata.obsm[self.embedding_key] = sc.tl.umap(
                adata, copy=True, method=method, **self.embedding_dict
            ).obsm["X_umap"]

import logging

import numpy as np
import scanpy as sc
import scvi
import torch


class SCANVI_POPV:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        n_epochs_unsupervised: int | None = None,
        n_epochs_semisupervised: int | None = None,
        save_folder: str | None = None,
        result_key: str | None = "popv_scanvi_prediction",
        embedding_key: str | None = "X_scanvi_umap_popv",
        model_kwargs: dict | None = None,
        classifier_kwargs: dict | None = None,
        embedding_dict: dict | None = None,
    ) -> None:
        """
        Class to compute classifier in scANVI model and predict labels.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
        labels_key
            Key in obs field of adata for cell-type information.
        n_epochs_unsupervised
            Number of epochs scvi is trained in unsupervised mode.
        n_epochs_semisupervised
            Number of epochs scvi is trained in semisupervised mode.
        result_key
            Key in obs in which celltype annotation results are stored.
        embedding_key
            Key in obsm in which UMAP embedding of integrated data is stored.
        model_kwargs
            Dictionary to supply non-default values for SCVI model. Options at scvi.model.SCVI
        classifier_kwargs
            Dictionary to supply non-default values for SCANVI classifier.
            Options at classifier_paramerers in scvi.model.SCANVI.from_scvi_model.
        embedding_dict
            Dictionary to supply non-default values for UMAP embedding. Options at sc.tl.umap
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.result_key = result_key
        self.embedding_key = embedding_key

        self.n_epochs_unsupervised = n_epochs_unsupervised
        self.n_epochs_semisupervised = n_epochs_semisupervised
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

        self.classifier_kwargs = {"n_layers": 3, "dropout_rate": 0.1}
        if classifier_kwargs is not None:
            self.classifier_kwargs.update(classifier_kwargs)

        self.embedding_dict = {"min_dist": 0.3}
        if embedding_dict is not None:
            self.embedding_dict.update(embedding_dict)

    def compute_integration(self, adata):
        logging.info("Integrating data with scANVI")

        # Go through obs field with subsampling information and subsample label information.
        if "subsampled_labels" not in adata.obs.columns:
            adata.obs["subsampled_labels"] = [
                label if subsampled else adata.uns["unknown_celltype_label"]
                for label, subsampled in zip(adata.obs["_labels_annotation"], adata.obs["_ref_subsample"])
            ]
        adata.obs["subsampled_labels"] = adata.obs["subsampled_labels"].astype("category")
        yprior = torch.tensor(
            [
                adata.obs["_labels_annotation"].value_counts()[i] / adata.n_obs
                for i in adata.obs["subsampled_labels"].cat.categories
                if i is not adata.uns["unknown_celltype_label"]
            ]
        )

        if self.n_epochs_unsupervised is None:
            self.n_epochs_unsupervised = round(min(round((10000 / adata.n_obs) * 200), 200))

        if adata.uns["_prediction_mode"] == "retrain":
            if adata.uns["_pretrained_scvi_path"] is not None:
                scvi_model = scvi.model.SCVI.load(adata.uns["_save_path_trained_models"] + "/scvi", adata=adata)
            else:
                scvi.model.SCVI.setup_anndata(
                    adata,
                    batch_key=self.batch_key,
                    labels_key="subsampled_labels",
                    layer="scvi_counts",
                )
                scvi_model = scvi.model.SCVI(adata, **self.model_kwargs)
                scvi_model.train(
                    train_size=1.0,
                    max_epochs=self.n_epochs_unsupervised,
                    accelerator=adata.uns["_accelerator"],
                    devices=adata.uns["_devices"],
                    plan_kwargs={"n_epochs_kl_warmup": 20},
                )

            self.model = scvi.model.SCANVI.from_scvi_model(
                scvi_model,
                unlabeled_category=adata.uns["unknown_celltype_label"],
                classifier_parameters=self.classifier_kwargs,
                y_prior=yprior,
            )
        else:
            query = adata[adata.obs["_dataset"] == "query"].copy()
            self.model = scvi.model.SCANVI.load_query_data(
                query,
                adata.uns["_save_path_trained_models"] + "/scanvi",
                freeze_classifier=True,
            )

        if adata.uns["_prediction_mode"] == "fast":
            if self.n_epochs_semisupervised is None:
                self.n_epochs_semisupervised = 1
            self.model.train(
                max_epochs=1,
                batch_size=512,
                n_samples_per_label=20,
                train_size=1.0,
                accelerator=adata.uns["_accelerator"],
                devices=adata.uns["_devices"],
                plan_kwargs={"n_steps_kl_warmup": 1},
            )
        else:
            if self.n_epochs_semisupervised is None:
                self.n_epochs_semisupervised = 20
            self.model.train(
                max_epochs=self.n_epochs_semisupervised,
                batch_size=512,
                n_samples_per_label=20,
                train_size=1.0,
                accelerator=adata.uns["_accelerator"],
                devices=adata.uns["_devices"],
                plan_kwargs={"n_epochs_kl_warmup": 20},
            )
        if adata.uns["_prediction_mode"] == "retrain":
            if adata.uns["_save_path_trained_models"] is not None:
                self.model.save(
                    adata.uns["_save_path_trained_models"] + "/scanvi",
                    save_anndata=False,
                    overwrite=True,
                )

    def predict(self, adata):
        logging.info(f'Saving scanvi label prediction to adata.obs["{self.result_key}"]')

        adata.obs[self.result_key] = self.model.predict(adata)
        if adata.uns["_return_probabilities"]:
            adata.obs[self.result_key + "_probabilities"] = np.max(self.model.predict(adata, soft=True), axis=1)

    def compute_embedding(self, adata):
        if adata.uns["_compute_embedding"]:
            logging.info(f'Saving UMAP of scanvi results to adata.obs["{self.embedding_key}"]')
            adata.obsm["X_scanvi"] = self.model.get_latent_representation(adata)
            sc.pp.neighbors(adata, use_rep="X_scanvi")
            adata.obsm[self.embedding_key] = sc.tl.umap(adata, copy=True, **self.embedding_dict).obsm["X_umap"]

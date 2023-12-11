import logging
import os
import subprocess
from typing import Union

import anndata
import numpy as np
import scanpy as sc
import torch
from scanpy._utils import check_nonnegative_integers

from popv import _utils


class Process_Query:
    """Class to preprocess AnnData for PopV."""

    def __init__(
        self,
        query_adata: anndata.AnnData,
        ref_adata: anndata.AnnData,
        ref_labels_key: str = "cell_ontology_class",
        ref_batch_key: str = "donor_method",
        query_labels_key: str | None = None,
        query_batch_key: str | None = None,
        query_layers_key: str | None = None,
        prediction_mode: str | None = "retrain",
        cl_obo_folder: Union[list, str, bool] | None = None,
        unknown_celltype_label: str | None = "unknown",
        n_samples_per_label: int | None = 300,
        pretrained_scvi_path: str | None = None,
        save_path_trained_models: str | None = "tmp/",
        hvg: int | None = 4000,
        accelerator: str | None = "cuda",
        devices: Union[int, str] | None = "auto",
        compute_embedding: bool = True,
        return_probabilities: bool = True,
    ) -> None:
        """
        Processes the query and reference dataset in preperation for the annotation pipeline.

        Parameters
        ----------
        query_adata
            AnnData of query cells
        ref_adata
            AnnData of reference cells
        ref_labels_key
            Key in obs field of reference AnnData with cell-type information
        ref_batch_keys
            List of Keys (or None) in obs field of reference AnnData to
            use as batch covariate
        query_labels_key
            Key in obs field of query adata for label information.
            This is only used for training scANVI.
            Make sure to set unknown_celltype_label to mark unlabelled cells.
        query_batch_key
            Key in obs field of query adata for batch information.
        query_layers_key
            If not None, expects raw_count data in query_layers_key.
        prediction_mode
            Execution mode of cell-type annotation.
            "retrain": Train all prediction models and saves them to disk if save_path_trained_models is not None.
            "inference": Classify all cells based on pretrained models.
            "fast": Fast inference using only query cells and single epoch in scArches.
        cl_obo_folder
            Folder containing the cell-type obo for Onclass, ontologies for Onclass and nlp embedding of cell-types.
            Passing a list will use element 1 as obo, element 2 as ontologies and element 3 as nlp embedding.
            Setting it to false will disable ontology use.
        unknown_celltype_label
            If query_labels_key is not None, cells with label unknown_celltype_label
            will be treated as unknown and will be predicted by the model.
        n_samples_per_label
            Reference AnnData will be subset to these amount of cells per cell-type to increase speed.
        pretrained_scvi_path
            If path is None, will train scVI from scratch. Else if
            pretrained_path is set and all the genes in the pretrained models are present
            in query adata, will train the scARCHES version of scVI and scANVI, resulting in
            faster training times.
        save_path_trained_models
            If mode=='retrain' saves models to this directory. Otherwise trained models are expected in this folder.
        hvg
            If Int, subsets data to n highly variable genes according to `sc.pp.highly_variable_genes`
        accelerator
            If using GPU, set accelerator to "cuda". If using CPU, set accelerator to "cpu".
        devices
            If using GPU, set devices to the GPU number. If using CPU, set devices to number of CPUs.
        compute_embedding
            Whether UMAP is computed for all integration methods (BBKNN, SCANORAMA, SCANVI, SCVI)
        return_probabilities
            Reports probabilities of the PopV prediction for each method where applicable
        """
        self.labels_key = {"reference": ref_labels_key, "query": query_labels_key}
        self.unknown_celltype_label = unknown_celltype_label
        self.batch_key = {"reference": ref_batch_key, "query": query_batch_key}

        if pretrained_scvi_path is None and prediction_mode != "retrain":
            self.pretrained_scvi_path = save_path_trained_models + "/scvi/"
        else:
            self.pretrained_scvi_path = pretrained_scvi_path

        if save_path_trained_models is not None:
            if save_path_trained_models[-1] != "/":
                save_path_trained_models += "/"
            if not os.path.exists(save_path_trained_models):
                os.makedirs(save_path_trained_models)
        self.save_path_trained_models = save_path_trained_models

        self.prediction_mode = prediction_mode
        self.return_probabilities = return_probabilities
        self.genes = None
        if self.prediction_mode == "fast":
            self.genes = torch.load(
                self.pretrained_scvi_path + "model.pt",
                map_location="cpu",
            )["var_names"]
        else:
            if self.pretrained_scvi_path is not None:
                pretrained_scvi_genes = torch.load(
                    self.pretrained_scvi_path + "model.pt",
                    map_location="cpu",
                )["var_names"]
                if self.prediction_mode == "inference":
                    pretrained_scanvi_genes = torch.load(
                        self.save_path_trained_models + "/scanvi/model.pt",
                        map_location="cpu",
                    )["var_names"]
                    assert list(pretrained_scvi_genes) == list(
                        pretrained_scanvi_genes
                    ), "Pretrained SCANVI and SCVI model contain different genes. This is not supported. Check models and retrain."

                    onclass_model = np.load(
                        self.save_path_trained_models + "/OnClass.npz",
                        allow_pickle=True,
                    )
                    assert set(onclass_model["genes"]).issubset(
                        set(pretrained_scanvi_genes)
                    ), "Pretrained SCANVI and OnClass model contain different genes. This is not supported. Retrain OnClass."
                else:
                    if not os.path.exists(self.save_path_trained_models):
                        os.makedirs(self.save_path_trained_models)
                self.genes = list(pretrained_scvi_genes)

        if self.genes is not None:
            assert set(self.genes).issubset(
                set(query_adata.var_names)
            ), "Query dataset misses genes that were used for reference model training. Retrain reference model, set mode='retrain'"
            self.query_adata = query_adata[:, self.genes].copy()
            assert hvg is None, "Highly variable gene selection is not available if using trained reference model."
        else:
            self.query_adata = query_adata.copy()
        if query_layers_key is not None:
            self.query_adata.X = self.query_adata.layers[query_layers_key].copy()

        self.validity_checked = False
        self.hvg = hvg
        self.accelerator = accelerator
        self.devices = devices
        if self.prediction_mode == "fast":
            self.n_samples_per_label = None
        else:
            self.n_samples_per_label = n_samples_per_label
        self.compute_embedding = compute_embedding

        if cl_obo_folder is None:
            self.cl_obo_file = os.path.dirname(os.path.dirname(__file__)) + "/ontology/cl.obo"
            self.cl_ontology_file = os.path.dirname(os.path.dirname(__file__)) + "/ontology/cl.ontology"
            self.nlp_emb_file = os.path.dirname(os.path.dirname(__file__)) + "/ontology/cl.ontology.nlp.emb"
            if not os.path.exists(self.nlp_emb_file):
                subprocess.call(
                    [
                        "tar",
                        "-czf",
                        os.path.dirname(os.path.dirname(__file__)) + "/ontology/nlp.emb.tar.gz",
                        "cl.ontology.nlp.emb",
                    ]
                )
        elif cl_obo_folder is False:
            self.cl_obo_file = False
            self.cl_ontology_file = False
            self.nlp_emb_file = False
        elif cl_obo_folder is list:
            self.cl_obo_file = cl_obo_folder[0]
            self.cl_ontology_file = cl_obo_folder[1]
            self.nlp_emb_file = cl_obo_folder[2]
        else:
            self.cl_obo_file = cl_obo_folder + "cl.obo"
            self.cl_ontology_file = cl_obo_folder + "cl.ontology"
            self.nlp_emb_file = cl_obo_folder + "cl.ontology.nlp.emb"
        if self.cl_obo_file:
            try:
                with open(self.cl_obo_file):
                    pass
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{self.cl_obo_file} doesn't exist. Check that folder exists."
                ) from FileNotFoundError

        self._check_validity_anndata(self.query_adata, "query")
        self._setup_dataset(self.query_adata, "query", add_meta="_query")

        if self.prediction_mode != "fast":
            if self.genes:
                self.ref_adata = ref_adata[:, self.genes].copy()
            else:
                self.ref_adata = ref_adata.copy()
            self._setup_dataset(self.ref_adata, "reference")
            self._check_validity_anndata(self.ref_adata, "reference")

        self._preprocess()

    def _check_validity_anndata(self, adata, input_type):
        assert check_nonnegative_integers(adata.X), f"Make sure input {input_type} adata contains raw_counts"
        assert len(set(adata.var_names)) == len(
            adata.var_names
        ), f"{input_type} dataset contains multiple genes with same gene name."
        assert adata.n_obs > 0, f"{input_type} anndata has no cells."
        assert adata.n_vars > 0, f"{input_type} anndata has no genes."

    def _setup_dataset(self, adata, key, add_meta=""):
        if isinstance(self.batch_key[key], list):
            adata.obs["_batch_annotation"] = adata.obs[self.batch_key[key]].astype(str).sum(1).astype("category")
        elif isinstance(self.batch_key[key], str):
            adata.obs["_batch_annotation"] = adata.obs[self.batch_key[key]]
        else:
            adata.obs["_batch_annotation"] = self.unknown_celltype_label
        adata.obs["_batch_annotation"] = adata.obs["_batch_annotation"].astype(str) + add_meta
        adata.obs["_batch_annotation"] = adata.obs["_batch_annotation"].astype("category")

        adata.obs["_labels_annotation"] = self.unknown_celltype_label
        if self.labels_key[key] is not None:
            adata.obs["_labels_annotation"] = adata.obs[self.labels_key[key]].astype("category")

        # subsample the reference cells used for training certain models
        if key == "reference":
            if self.n_samples_per_label is not None:
                adata.obs["_ref_subsample"] = False
                subsample_idx = _utils.subsample_dataset(
                    adata,
                    self.labels_key[key],
                    n_samples_per_label=self.n_samples_per_label,
                    ignore_label=[self.unknown_celltype_label],
                )
                adata.obs.loc[subsample_idx, "_ref_subsample"] = True
            else:
                adata.obs["_ref_subsample"] = True
        else:
            adata.obs["_ref_subsample"] = False

    def _preprocess(self):
        if self.genes is None:
            self.ref_adata = self.ref_adata[:, np.intersect1d(self.ref_adata.var_names, self.query_adata.var_names)]
            self.query_adata = self.query_adata[:, np.intersect1d(self.ref_adata.var_names, self.query_adata.var_names)]

        if self.prediction_mode == "fast":
            self.adata = self.query_adata
            self.adata.obs["_dataset"] = "query"
        else:
            self.adata = anndata.concat(
                (self.ref_adata, self.query_adata),
                axis=0,
                label="_dataset",
                keys=["ref", "query"],
                join="outer",
                fill_value=self.unknown_celltype_label,
            )

        if self.prediction_mode != "fast":
            # Necessary for BBKNN.
            batch_before_filtering = set(self.adata.obs["_batch_annotation"])
            self.adata = self.adata[
                self.adata.obs["_batch_annotation"].isin(
                    self.adata.obs["_batch_annotation"]
                    .value_counts()[self.adata.obs["_batch_annotation"].value_counts() > 8]
                    .index
                )
            ].copy()
            difference_batches = set(self.adata.obs["_batch_annotation"]) - batch_before_filtering
            if difference_batches:
                logging.warning(
                    f"The following batches will be excluded from annotation because they have less than 9 cells:{difference_batches}."
                )

            # Sort data based on batch for efficiency downstream during SCANORAMA
            self.adata = self.adata[self.adata.obs.sort_values(by="_batch_annotation").index].copy()

            self.adata.obs[self.labels_key["reference"]] = self.adata.obs[self.labels_key["reference"]].astype(
                "category"
            )

        # Remove any cell with expression below 10 counts.
        zero_cell_names = self.adata[self.adata.X.sum(1) < 10].obs_names
        self.adata.uns["Filtered_cells"] = list(zero_cell_names)
        sc.pp.filter_cells(self.adata, min_counts=30, inplace=True)
        if len(zero_cell_names) > 0:
            logging.warning(
                f"The following cells will be excluded from annotation because they have low expression:{zero_cell_names}."
            )

        self.adata.layers["scvi_counts"] = self.adata.X.copy()

        if self.hvg is not None and self.adata.n_vars > self.hvg:
            sc.pp.filter_genes(self.adata, min_counts=20, inplace=True)
            try:
                self.adata.var["highly_variable"] = sc.pp.highly_variable_genes(
                    self.adata[self.adata.obs["_dataset"] == "ref"].copy(),
                    n_top_genes=self.hvg,
                    subset=False,
                    layer="scvi_counts",
                    flavor="seurat_v3",
                    inplace=False,
                    batch_key="_batch_annotation",
                )["highly_variable"]
            except ValueError:  # seurat_v3 tends to error with singularities then use Poisson hvg.
                self.adata.var["highly_variable"] = sc.experimental.pp.highly_variable_genes(
                    self.adata[self.adata.obs["_dataset"] == "ref"].copy(),
                    n_top_genes=self.hvg,
                    subset=False,
                    layer="scvi_counts",
                    flavor="pearson_residuals",
                    inplace=False,
                    batch_key="_batch_annotation",
                )["highly_variable"]
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.adata.layers["scaled_counts"] = self.adata.X.copy()
        if self.prediction_mode != "fast":
            sc.pp.scale(self.adata, max_value=10, zero_center=False, layer="scaled_counts")
            self.adata.obsm["X_pca"] = sc.tl.pca(self.adata.layers["scaled_counts"])

        # Store values as default for current popv in adata
        self.adata.uns["unknown_celltype_label"] = self.unknown_celltype_label
        self.adata.uns["_pretrained_scvi_path"] = self.pretrained_scvi_path
        self.adata.uns["_save_path_trained_models"] = self.save_path_trained_models
        self.adata.uns["_prediction_mode"] = self.prediction_mode
        self.adata.uns["_cl_obo_file"] = self.cl_obo_file
        self.adata.uns["_cl_ontology_file"] = self.cl_ontology_file
        self.adata.uns["_nlp_emb_file"] = self.nlp_emb_file
        self.adata.uns["_accelerator"] = self.accelerator
        self.adata.uns["_devices"] = self.devices
        self.adata.uns["_compute_embedding"] = self.compute_embedding
        self.adata.uns["_return_probabilities"] = self.return_probabilities
        self.adata.uns["prediction_keys"] = []

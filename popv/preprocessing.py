import logging
import os
import subprocess
from typing import List, Optional, Union

import anndata
import numpy as np
import scanpy as sc
import torch
from scanpy._utils import check_nonnegative_integers

from popv import _utils
from popv import settings


class Process_Query:
    def __init__(
        self,
        query_adata: anndata.AnnData,
        ref_adata: anndata.AnnData,
        ref_labels_key: str,
        ref_batch_key: str,
        query_labels_key: Optional[str] = None,
        query_batch_key: Optional[str] = None,
        query_layers_key: Optional[str] = None,
        prediction_mode: Optional[str] = "retrain",
        cl_obo_folder: Optional[Union[List, str, bool]] = None,
        unknown_celltype_label: Optional[str] = "unknown",
        n_samples_per_label: Optional[int] = 300,
        pretrained_scvi_path: Optional[str] = False,
        save_path_trained_models: Optional[str] = "tmp/",
        hvg: Optional[int] = 4000,
        use_gpu: Optional[bool] = True,
        compute_embedding: Optional[bool] = True,
        return_probabilities: Optional[bool] = True,
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
        use_gpu
            If gpu is used to train scVI and scANVI model
        compute_embedding
            Whether UMAP is computed for all integration methods (BBKNN, SCANORAMA, SCANVI, SCVI)
        return_probabilities
            Reports probabilities of the PopV prediction for each method where applicable
        """

        self.labels_key = {"reference": ref_labels_key, "query": query_labels_key}
        self.unknown_celltype_label = unknown_celltype_label
        self.batch_key = {"reference": ref_batch_key, "query": query_batch_key}

        if pretrained_scvi_path and prediction_mode != "retrain":
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
            assert self.pretrained_scvi_path, 'Fast mode requires a pretrained scvi model.'
            self.genes = torch.load(
                self.pretrained_scvi_path + "model.pt",
                map_location="cpu",
            )["var_names"]
        else:
            if self.pretrained_scvi_path:
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
            assert (
                hvg is None
            ), "Highly variable gene selection is not available if using trained reference model."
        else:
            gene_intersection = np.intersect1d(ref_adata.var_names, query_adata.var_names)
            if hvg is not None and len(gene_intersection) > hvg:
                expressed_genes, _ = sc.pp.filter_genes(
                    query_adata[:, gene_intersection], min_cells=200, inplace=False)
                subset_genes = gene_intersection[expressed_genes]
                highly_variable_genes = sc.pp.highly_variable_genes(
                    query_adata[:, subset_genes].copy(),
                    n_top_genes=hvg,
                    subset=False,
                    flavor="seurat_v3",
                    inplace=False,
                    layer=query_layers_key,
                    batch_key=query_batch_key,
                    span=1.0,
                )["highly_variable"]
                self.genes = query_adata[:, subset_genes].var_names[highly_variable_genes]
            else:
                self.genes = gene_intersection
        self.query_adata = query_adata[:, self.genes].copy()
        if query_layers_key is not None:
            self.query_adata.X = self.query_adata.layers[query_layers_key].copy()

        self.validity_checked = False
        self.use_gpu = use_gpu
        if self.prediction_mode == "fast":
            self.n_samples_per_label = None
        else:
            self.n_samples_per_label = n_samples_per_label
        self.compute_embedding = compute_embedding

        if cl_obo_folder is None:
            self.cl_obo_file = (
                os.path.dirname(os.path.dirname(__file__)) + "/ontology/cl.obo"
            )
            self.cl_ontology_file = (
                os.path.dirname(os.path.dirname(__file__)) + "/ontology/cl.ontology"
            )
            self.nlp_emb_file = (
                os.path.dirname(os.path.dirname(__file__))
                + "/ontology/cl.ontology.nlp.emb"
            )
            if not os.path.exists(self.nlp_emb_file):
                subprocess.call(
                    [
                        "tar",
                        "-czf",
                        os.path.dirname(os.path.dirname(__file__))
                        + "/ontology/nlp.emb.tar.gz",
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
                )

        self.setup_dataset(self.query_adata, "query")
        self.check_validity_anndata(self.query_adata, "query")
        self.setup_dataset(self.query_adata, "query", add_meta="_query")

        if self.prediction_mode != "fast":
            self.ref_adata = ref_adata[:, self.genes].copy()
            self.setup_dataset(self.ref_adata, "reference")
            self.check_validity_anndata(self.ref_adata, "reference")

        self.preprocess()

    def check_validity_anndata(self, adata, input_type):
        assert check_nonnegative_integers(
            adata.X
        ), f"Make sure input {input_type} adata contains raw_counts"
        assert len(set(adata.var_names)) == len(
            adata.var_names
        ), f"{input_type} dataset contains multiple genes with same gene name."
        assert adata.n_obs > 0, f"{input_type} anndata has no cells."
        assert adata.n_vars > 0, f"{input_type} anndata has no genes."

    def setup_dataset(self, adata, key, add_meta=""):
        if isinstance(self.batch_key[key], list):
            adata.obs["_batch_annotation"] = (
                adata.obs[self.batch_key[key]].astype(str).sum(1).astype("category")
            )
        elif isinstance(self.batch_key[key], str):
            adata.obs["_batch_annotation"] = adata.obs[self.batch_key[key]]
        else:
            adata.obs["_batch_annotation"] = self.unknown_celltype_label
        adata.obs["_batch_annotation"] = (
            adata.obs["_batch_annotation"].astype(str) + add_meta
        )
        adata.obs["_batch_annotation"] = adata.obs["_batch_annotation"].astype(
            "category"
        )

        adata.obs["_labels_annotation"] = self.unknown_celltype_label
        if self.labels_key[key] is not None:
            adata.obs["_labels_annotation"] = adata.obs[self.labels_key[key]].astype(
                "category"
            )

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

    def preprocess(self):

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
        del self.query_adata, self.ref_adata
        self.adata.obs["_labelled_train_indices"] = np.logical_and(
            self.adata.obs["_dataset"]=="ref",
            self.adata.obs["_labels_annotation"]!=self.unknown_celltype_label)

        if self.prediction_mode != "fast":
            # Necessary for BBKNN.
            batch_before_filtering = set(self.adata.obs["_batch_annotation"])
            self.adata = self.adata[
                self.adata.obs["_batch_annotation"].isin(
                    self.adata.obs["_batch_annotation"]
                    .value_counts()[
                        self.adata.obs["_batch_annotation"].value_counts() > 8
                    ]
                    .index
                )
            ]
            difference_batches = (
                set(self.adata.obs["_batch_annotation"]) - batch_before_filtering
            )
            if difference_batches:
                logging.warning(
                    f"The following batches will be excluded from annotation because they have less than 9 cells:{difference_batches}."
                )

            # Sort data based on batch for efficiency downstream during SCANORAMA
            self.adata = self.adata[
                self.adata.obs.sort_values(by="_batch_annotation").index
            ]

            self.adata.obs[self.labels_key["reference"]] = self.adata.obs[
                self.labels_key["reference"]
            ].astype("category")

        # Remove any cell with expression below 10 counts.
        zero_cell_names = self.adata[self.adata.X.sum(1) < 10].obs_names
        self.adata.uns["Filtered_cells"] = list(zero_cell_names)
        sc.pp.filter_cells(self.adata, min_counts=30, inplace=True)
        if len(zero_cell_names) > 0:
            logging.warning(
                f"The following cells will be excluded from annotation because they have low expression:{zero_cell_names}."
            )
        self.adata.layers["scvi_counts"] = self.adata.X.copy()

        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.adata.layers["scaled_counts"] = self.adata.X.copy()
        if self.prediction_mode != "fast":
            sc.pp.scale(
                self.adata, max_value=10, zero_center=False, layer="scaled_counts"
            )
            self.adata.obsm["X_pca"] = sc.tl.pca(self.adata.layers["scaled_counts"])

        self.adata.obs["_labels_annotation"] = self.adata.obs["_labels_annotation"].astype('category')
        # Store values as default for current popv in adata
        self.adata.uns["unknown_celltype_label"] = self.unknown_celltype_label
        self.adata.uns["_pretrained_scvi_path"] = self.pretrained_scvi_path
        self.adata.uns["_save_path_trained_models"] = self.save_path_trained_models
        self.adata.uns["_prediction_mode"] = self.prediction_mode
        self.adata.uns["_cl_obo_file"] = self.cl_obo_file
        self.adata.uns["_cl_ontology_file"] = self.cl_ontology_file
        self.adata.uns["_nlp_emb_file"] = self.nlp_emb_file
        self.adata.uns["_use_gpu"] = self.use_gpu
        self.adata.uns["_compute_embedding"] = self.compute_embedding
        self.adata.uns["_return_probabilities"] = self.return_probabilities
        self.adata.uns["prediction_keys"] = []

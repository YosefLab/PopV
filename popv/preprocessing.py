import logging
from typing import Optional

import anndata
import numpy as np
import scanpy as sc
import scvi
from scanpy._utils import check_nonnegative_integers

from popv import _utils


class Process_Query:
    def __init__(
        self,
        query_adata: anndata.AnnData,
        ref_adata: anndata.AnnData,
        ref_labels_key: str,
        cl_obo_file: Optional[str] = "ontology/cl.obo",
        cl_ontology_file: Optional[str] = "ontology/cl.ontology",
        nlp_emb_file: Optional[str] = "ontology/cl.ontology.nlp.emb",
        query_labels_key: Optional[str] = None,
        unknown_celltype_label: Optional[str] = "unknown",
        n_samples_per_label: Optional[int] = 100,
        ref_batch_key: str = "donor_method",
        query_batch_key: Optional[str] = None,
        save_folder: str = "results_popv",
        query_layers_key=None,
        pretrained_scvi_path: Optional[str] = None,
        pretrained_scanvi_path: Optional[str] = None,
        hvg: Optional[int] = 4000,
        use_gpu: Optional[bool] = False,
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
        cl_obo_file
            File containing the cell-type obo for Onclass.
        cl_ontology_file
            File containing the cell-type ontologies for Onclass.
        nlp_emb_file
            File containing nlp embedding of cell-types.
        query_labels_key
            Key in obs field of query adata for label information.
            This is only used for training scANVI.
            Make sure to set unknown_celltype_label to mark unlabelled cells.
        unknown_celltype_label
            If query_labels_key is not None, cells with label unknown_celltype_label
            will be treated as unknown and will be predicted by the model.
        n_samples_per_label
            Reference AnnData will be subset to these amount of cells per cell-type to increase speed.
        ref_batch_keys
            List of Keys (or None) in obs field of reference AnnData to
            use as batch covariate
        query_batch_key
            Key in obs field of query adata for batch information.
        save_folder
            Folder to save data to
        query_layers_key
            If not None, will use data from query_adata.layers[query_layers_key].
        pretrained_scvi_path
            If path is None, will train scVI from scratch. Else if
            pretrained_path is set and all the genes in the pretrained models are present
            in query adata, will train the scARCHES version of scVI and scANVI, resulting in
            faster training times.
        pretrained_scanvi_path
            If path is None, will train scANVI from scratch. Else if
            pretrained_path is set and all the genes in the pretrained models are present
            in query adata, will train the scARCHES version of scVI and scANVI, resulting in
            faster training times.
        hvg
            If Int, subsets data to n highly variable genes according to `sc.pp.highly_variable_genes`
        use_gpu
            If gpu is used to train scVI and scANVI model
        """

        self.query_adata = query_adata.copy()
        if query_layers_key is not None:
            self.query_adata.X = self.query_adata.layers[query_layers_key].copy()
        self.ref_adata = ref_adata.copy()
        self.validity_checked = False

        self.labels_key = {"reference": ref_labels_key, "query": query_labels_key}
        self.unknown_celltype_label = unknown_celltype_label
        self.n_samples_per_label = n_samples_per_label

        self.batch_key = {"reference": ref_batch_key, "query": query_batch_key}
        self.dataset_prepared = False

        self.save_folder = save_folder
        self.pretrained_scvi_path = pretrained_scvi_path
        self.pretrained_scanvi_path = pretrained_scanvi_path
        if pretrained_scvi_path is not None and pretrained_scanvi_path is not None:
            pretrained_data = scvi.model.SCVI.load(self.pretrained_scvi_path).adata
            pretrained_data_scanvi = scvi.model.SCANVI.load(
                self.pretrained_scvi_path
            ).adata
            assert (
                pretrained_data.var_names == pretrained_data_scanvi.var_names
            ), "Pretrained SCANVI and SCVI model contain different genes"
            assert (
                pretrained_data.obs_names == pretrained_data_scanvi.obs_names
            ), "Pretrained SCANVI and SCVI model contain different cells"
        self.hvg = hvg
        self.use_gpu = use_gpu

        self.cl_obo_file = cl_obo_file
        self.cl_ontology_file = cl_ontology_file
        self.nlp_emb_file = nlp_emb_file

        self.check_validity_anndata(ref_adata, "reference")
        self.check_validity_anndata(query_adata, "query")

        self.setup_dataset(self.query_adata, "query", add_meta="_query")
        self.setup_dataset(self.ref_adata, "reference")

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
        if self.pretrained_scvi_path is not None:
            pretrained_data = scvi.model.SCVI.load(self.pretrained_scvi_path).adata
            adata[:, pretrained_data.var_names].copy()
            assert (
                self.hvg is None
            ), "Highly variable gene selection is not available if using trained reference model."
            assert (
                adata.var_names == pretrained_data.var_names
            ), "Query dataset misses genes that were used for reference model training. Retrain reference model or set online=False"

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
            adata.obs["_labels_annotation"] = adata.obs[self.labels_key[key]]

        # subsample the reference cells used for training certain models
        adata.obs["_ref_subsample"] = False
        if self.n_samples_per_label is not None and key == "reference":
            subsample_idx = _utils.subsample_dataset(
                adata,
                self.labels_key[key],
                n_samples_per_label=self.n_samples_per_label,
                ignore_label=[self.unknown_celltype_label],
            )
            adata.obs.loc[subsample_idx, "_ref_subsample"] = True

    def preprocess(self):
        self.ref_adata = self.ref_adata[
            :, np.intersect1d(self.ref_adata.var_names, self.query_adata.var_names)
        ]
        self.query_adata = self.query_adata[
            :, np.intersect1d(self.ref_adata.var_names, self.query_adata.var_names)
        ]

        self.adata = anndata.concat(
            (self.ref_adata, self.query_adata),
            axis=0,
            label="_dataset",
            keys=["ref", "query"],
            join="outer",
            fill_value=self.unknown_celltype_label,
        )
        self.adata.obs[self.labels_key["reference"]] = self.adata.obs[self.labels_key["reference"]].astype('category')

        self.adata.obs[self.labels_key["reference"]] = (
            self.adata.obs[self.labels_key["reference"]]
            .cat.add_categories("unknown")
            .fillna("unknown")
        )

        # Remove any cell with expression below 10 counts.
        zero_cell_names = self.adata[self.adata.X.sum(1) < 10].obs_names
        self.adata.uns["Filtered_cells"] = zero_cell_names
        sc.pp.filter_cells(self.adata, min_counts=10, inplace=True)
        if len(zero_cell_names) > 0:
            logging.warning(
                f"The following cells will be excluded from annotation because they have low expression:{zero_cell_names}, likely due to highly variable gene selection. We recommend you subset the data yourself and set hvg to False."
            )

        self.adata.layers["scvi_counts"] = self.adata.X.copy()
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.adata.layers["logcounts"] = self.adata.X.copy()

        if self.hvg is not None:
            n_top_genes = np.min([self.hvg, self.adata.n_vars])
            sc.pp.highly_variable_genes(
                self.adata[self.adata.obs["_dataset"] == "ref"],
                n_top_genes=n_top_genes,
                subset=True,
                layer="scvi_counts",
                flavor="seurat",  # CAN switch back to seurat_v3 and add skmisc to dependencies
            )
        sc.pp.scale(self.adata, max_value=10, zero_center=False)
        sc.tl.pca(self.adata)

        # Store values as default for current popv in adata
        self.adata.uns["unknown_celltype_label"] = self.unknown_celltype_label
        self.adata.uns["_pretrained_scvi_path"] = self.pretrained_scvi_path
        self.adata.uns["_pretrained_scanvi_path"] = self.pretrained_scanvi_path
        self.adata.uns["_cl_obo_file"] = self.cl_obo_file
        self.adata.uns["_cl_ontology_file"] = self.cl_ontology_file
        self.adata.uns["_nlp_emb_file"] = self.nlp_emb_file
        self.adata.uns["_use_gpu"] = self.use_gpu

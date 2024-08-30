from __future__ import annotations

from abc import abstractmethod

from popv import settings


class BaseAlgorithm:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        seen_result_key: str | None = None,
        result_key: str | None = None,
        embedding_key: str | None = None,
        layer_key: str | None = None,
    ) -> None:
        """
        Class to compute KNN classifier after BBKNN integration.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
        labels_key
            Key in obs field of adata for cell-type information.
        seen_result_key
            Key in obs in which seen celltype annotation results are stored.
        result_key
            Key in obs in which celltype annotation results are stored.
        embedding_key
            Key in obsm in which UMAP embedding of integrated data is stored.
        layer_key
            AnnData layer to use for celltype prediction.
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        if seen_result_key is None:
            self.seen_result_key = result_key
        else:
            self.seen_result_key = seen_result_key
        self.result_key = result_key
        self.embedding_key = embedding_key
        self.layer_key = layer_key
        self.enable_cuml = settings.cuml
        self.return_probabilities = settings.return_probabilities
        self.compute_embedding = settings.compute_embedding

    @abstractmethod
    def _compute_integration(self, adata):
        """Computes integration of adata"""

    @abstractmethod
    def _predict(self, adata):
        """Predicts cell type of adata"""

    @abstractmethod
    def _compute_embedding(self, adata):
        """Computes UMAP embedding of adata"""

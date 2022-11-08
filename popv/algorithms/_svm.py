from ast import Pass
import scanpy as sc
import numpy as np
import logging

from sklearn import svm
from typing import Optional, Literal


class SVM:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        layers_key: Optional[str] = "logcounts",
        result_key: Optional[str] = "popv_svm_prediction",
        embedding_key: Optional[str] = None,
        classifier_dict: Optional[str] = {},
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
            Here for consistency with other methods.
        classifier_dict
            Dictionary to supply non-default values for SVM classifier. Options at sklearn.svm.
        """

        self.batch_key = batch_key
        self.labels_key = labels_key
        self.layers_key = layers_key
        self.result_key = result_key
        self.embedding_key = embedding_key

        self.classifier_dict = {
            "C": 1,
            "max_iter": 5000,
            "class_weight": "balanced",
        }
        self.classifier_dict.update(classifier_dict)
        
    def compute_integration(self, adata):
        Pass
    
    def predict(self, adata):
        logging.info('Computing support vector machine. Storing prediction in adata.obs["{}"]'.format(self.result_key))

        train_idx = adata.obs["_ref_subsample"]
        test_idx = adata.obs["_dataset"] == "query"

        train_x = adata[train_idx].layers[self.layers_key]
        train_y = adata[train_idx].obs[self.labels_key].to_numpy()
        test_x = adata[test_idx].layers[self.layers_key]

        clf = svm.LinearSVC(**self.classifier_dict)
        clf.fit(train_x, train_y)
        svm_pred = clf.predict(
            test_x,
        )

        adata.obs[self.result_key] = adata.obs[self.labels_key]
        adata.obs.loc[test_idx, self.result_key] = svm_pred

    def compute_embedding(self, adata):
        pass
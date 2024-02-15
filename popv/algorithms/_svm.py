import logging
import pickle
from ast import Pass
from typing import Optional
import pandas as pd
import scipy.sparse as scp

import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from popv import settings


class SVM:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        layers_key: Optional[str] = None,
        result_key: Optional[str] = "popv_svm_prediction",
        classifier_dict: Optional[str] = {},
    ) -> None:
        """
        Class to compute LinearSVC.

        Parameters
        ----------
        batch_key
            Key in obs field of adata for batch information.
        labels_key
            Key in obs field of adata for cell-type information.
        layers_key
            Key in layers field of adata used for classification. By default uses 'X' (log1p10K).
        result_key
            Key in obs in which celltype annotation results are stored.
        classifier_dict
            Dictionary to supply non-default values for SVM classifier. Options at sklearn.svm.
        """

        self.batch_key = batch_key
        self.labels_key = labels_key
        self.layers_key = layers_key
        self.result_key = result_key

        self.classifier_dict = {
            "C": 1,
            "max_iter": 5000,
            "class_weight": "balanced",
        }
        self.classifier_dict.update(classifier_dict)

    def compute_integration(self, adata):
        Pass

    def predict(self, adata):
        logging.info(
            'Computing support vector machine. Storing prediction in adata.obs["{}"]'.format(
                self.result_key
            )
        )
        test_x = adata.layers[self.layers_key] if self.layers_key else adata.X

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = (
                adata[train_idx].layers[self.layers_key]
                if self.layers_key
                else adata[train_idx].X
            )
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            if settings.cuml:
                from cuml.svm import LinearSVC
                from sklearn.multiclass import OneVsRestClassifier
                self.classifier_dict['probability'] = adata.uns["_return_probabilities"]
                clf = OneVsRestClassifier(LinearSVC(**self.classifier_dict))
                train_x = train_x.todense()
            else:
                clf = CalibratedClassifierCV(svm.LinearSVC(**self.classifier_dict))
            clf.fit(train_x, train_y)
            if adata.uns["_save_path_trained_models"] and not settings.cuml:
                pickle.dump(
                    clf,
                    open(
                        adata.uns["_save_path_trained_models"] + "svm_classifier.pkl",
                        "wb",
                    ),
                )
        else:
            clf = pickle.load(
                open(
                    adata.uns["_save_path_trained_models"] + "svm_classifier.pkl", "rb"
                )
            )

        if settings.cuml and scp.issparse(test_x):
            if adata.uns["_return_probabilities"]:
                required_columns = [
                    self.result_key, self.result_key + "_probabilities"]
            else:
                required_columns = [
                    self.result_key]

            result_df = pd.DataFrame(
                index=adata.obs_names,
                columns=required_columns
            )
            shard_size = int(settings.shard_size)
            for i in range(0, adata.n_obs, shard_size):
                tmp_x = test_x[i: i + shard_size]
                names_x = adata.obs_names[i: i + shard_size]
                tmp_x = tmp_x.todense()
                result_df.loc[names_x, self.result_key] = adata.obs[self.labels_key].cat.categories[clf.predict(tmp_x).astype(int)]
                if adata.uns["_return_probabilities"]:
                    result_df.loc[names_x, self.result_key + "_probabilities"] = np.max(
                        clf.predict_proba(tmp_x), axis=1
                    ).astype(float)
            adata.obs[result_df.columns] = result_df
        else:
            adata.obs[self.result_key] = adata.obs[self.labels_key].cat.categories[clf.predict(test_x)]
            if adata.uns["_return_probabilities"]:
                adata.obs[self.result_key + "_probabilities"] = np.max(
                    clf.predict_proba(test_x), axis=1
                )

    def compute_embedding(self, adata):
        pass

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as scp
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class SVM(BaseAlgorithm):
    """
    Class to compute LinearSVC.

    Parameters
    ----------
    batch_key
        Key in obs field of adata for batch information.
    labels_key
        Key in obs field of adata for cell-type information.
    layer_key
        Key in layers field of adata used for classification. By default uses 'X' (log1p10K).
    result_key
        Key in obs in which celltype annotation results are stored.
    classifier_dict
        Dictionary to supply non-default values for SVM classifier. Options at sklearn.svm.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_svm_prediction",
        classifier_dict: str | None = {},
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            layer_key=layer_key,
        )

        self.classifier_dict = {
            "C": 1,
            "max_iter": 5000,
            "class_weight": "balanced",
        }
        self.classifier_dict.update(classifier_dict)

    def _predict(self, adata):
        logging.info(
            f'Computing support vector machine. Storing prediction in adata.obs["{self.result_key}"]'
        )
        test_x = adata.layers[self.layer_key] if self.layer_key else adata.X

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = (
                adata[train_idx].layers[self.layer_key]
                if self.layer_key
                else adata[train_idx].X
            )
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            if settings.cuml:
                from cuml.svm import LinearSVC
                from sklearn.multiclass import OneVsRestClassifier
                self.classifier_dict['probability'] = self.return_probabilities
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
            if self.return_probabilities:
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
                if self.return_probabilities:
                    result_df.loc[names_x, self.result_key + "_probabilities"] = np.max(
                        clf.predict_proba(tmp_x), axis=1
                    ).astype(float)
            adata.obs[result_df.columns] = result_df
        else:
            adata.obs[self.result_key] = adata.obs[self.labels_key].cat.categories[clf.predict(test_x)]
            if self.return_probabilities:
                adata.obs[self.result_key + "_probabilities"] = np.max(
                    clf.predict_proba(test_x), axis=1
                )

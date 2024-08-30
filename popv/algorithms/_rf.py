from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as scp
from sklearn.ensemble import RandomForestClassifier

from popv import settings
from popv.algorithms._base_algorithm import BaseAlgorithm


class RF(BaseAlgorithm):
    """
    Class to compute Random forest classifier.

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
    enable_cuml
        Enable cuml, which currently doesn't support weighting. Default to popv.settings.cuml.
    classifier_dict
        Dictionary to supply non-default values for RF classifier. Options at sklearn.ensemble.RandomForestClassifier.
    """

    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layer_key: str | None = None,
        result_key: str | None = "popv_rf_prediction",
        classifier_dict: str | None = {},
    ) -> None:
        super().__init__(
            batch_key=batch_key,
            labels_key=labels_key,
            result_key=result_key,
            layer_key=layer_key,
        )

        self.classifier_dict = {
            "class_weight": "balanced_subsample",
            "max_features": 200,
            "n_jobs": settings.n_jobs,
        }
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def _predict(self, adata):
        logging.info(
            f'Computing random forest classifier. Storing prediction in adata.obs["{self.result_key}"]'
        )

        test_x = adata.layers[self.layer_key] if self.layer_key else adata.X

        if len(adata.obs[self.labels_key].unique())>100 and settings.cuml:
            logging.warning('cuml.ensemble.RandomForestClassifier leads to OOM for more than a hundred labels. Disabling cuML and using sklearn.')
            enable_cuml = False
        else:
            enable_cuml = settings.cuml

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = (
                adata[train_idx].layers[self.layer_key]
                if self.layer_key
                else adata[train_idx].X
            )
            train_y = adata.obs.loc[train_idx, self.labels_key].cat.codes.to_numpy()
            if enable_cuml:
                from cuml.ensemble import RandomForestClassifier as cuRF
                self.classifier_dict = {
                    "max_features": 200,
                }
                rf = cuRF(**self.classifier_dict)
                train_x = train_x.todense()
            else:
                rf = RandomForestClassifier(**self.classifier_dict)
            rf.fit(train_x, train_y)
            if adata.uns["_save_path_trained_models"] and not enable_cuml:
                pickle.dump(
                    rf,
                    open(
                        adata.uns["_save_path_trained_models"] + "rf_classifier.pkl",
                        "wb",
                    ),
                )
        else:
            rf = pickle.load(
                open(adata.uns["_save_path_trained_models"] + "rf_classifier.pkl", "rb")
            )

        if enable_cuml and scp.issparse(test_x):
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
                tmp_x = test_x[i : i + shard_size]
                names_x = adata.obs_names[i : i + shard_size]
                tmp_x = tmp_x.todense()
                result_df.loc[names_x, self.result_key] = adata.obs[self.labels_key].cat.categories[rf.predict(tmp_x, predict_model='CPU').astype(int)]
                if self.return_probabilities:
                    try:
                        result_df.loc[names_x, self.result_key + "_probabilities"] = np.max(
                            rf.predict_proba(tmp_x), axis=1
                        )
                    except MemoryError:
                        logging.warning(
                            "Memory error while computing probabilities. Disabling probabilities."
                        )
                        result_df.loc[names_x, self.result_key + "_probabilities"] = None

            adata.obs[result_df.columns] = result_df
        else:
            adata.obs[self.result_key] = adata.obs[self.labels_key].cat.categories[rf.predict(test_x)]
            if self.return_probabilities:
                adata.obs[self.result_key + "_probabilities"] = np.max(
                    rf.predict_proba(test_x), axis=1
                ).astype(float)

import logging
import pickle
from ast import Pass
from typing import Optional
import pandas as pd
import scipy.sparse as scp

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from popv import settings


class RF:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        layers_key: Optional[str] = None,
        result_key: Optional[str] = "popv_rf_prediction",
        classifier_dict: Optional[str] = {},
    ) -> None:
        """
        Class to compute Random forest classifier.

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
        enable_cuml
            Enable cuml, which currently doesn't support weighting. Default to popv.settings.cuml.
        classifier_dict
            Dictionary to supply non-default values for RF classifier. Options at sklearn.ensemble.RandomForestClassifier.
        """

        self.batch_key = batch_key
        self.labels_key = labels_key
        self.layers_key = layers_key
        self.result_key = result_key
        self.enable_cuml = settings.cuml

        self.classifier_dict = {
            "class_weight": "balanced_subsample",
            "max_features": 200,
            "n_jobs": settings.n_jobs,
        }
        self.classifier_dict.update(classifier_dict)

    def compute_integration(self, adata):
        Pass

    def predict(self, adata):
        logging.info(
            'Computing random forest classifier. Storing prediction in adata.obs["{}"]'.format(
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
            if len(adata.obs[self.labels_key].unique())>100 and settings.cuml:
                logging.warning('cuml.ensemble.RandomForestClassifier leads to OOM for more than a hundred labels. Disabling cuML and using sklearn.')
                enable_cuml = False
            else:
                enable_cuml = settings.cuml
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
                tmp_x = test_x[i : i + shard_size]
                names_x = adata.obs_names[i : i + shard_size]
                tmp_x = tmp_x.todense()
                result_df.loc[names_x, self.result_key] = adata.obs[self.labels_key].cat.categories[rf.predict(tmp_x, predict_model='CPU').astype(int)]
                if adata.uns["_return_probabilities"]:
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
            if adata.uns["_return_probabilities"]:
                adata.obs[self.result_key + "_probabilities"] = np.max(
                    rf.predict_proba(test_x), axis=1
                ).astype(float)

    def compute_embedding(self, adata):
        pass

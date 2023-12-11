import logging
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RF:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layers_key: str | None = None,
        result_key: str | None = "popv_rf_prediction",
        classifier_dict: str | None = {},
    ) -> None:
        """
        Class to compute KNN classifier after BBKNN integration.

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
            Dictionary to supply non-default values for RF classifier. Options at sklearn.ensemble.RandomForestClassifier.
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.layers_key = layers_key
        self.result_key = result_key

        self.classifier_dict = {
            "class_weight": "balanced_subsample",
            "max_features": 200,
        }
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def compute_integration(self, adata):
        pass

    def predict(self, adata):
        logging.info(f'Computing random forest classifier. Storing prediction in adata.obs["{self.result_key}"]')

        test_x = adata.layers[self.layers_key] if self.layers_key else adata.X

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = adata[train_idx].layers[self.layers_key] if self.layers_key else adata[train_idx].X
            train_y = adata[train_idx].obs[self.labels_key].to_numpy()
            rf = RandomForestClassifier(**self.classifier_dict)
            rf.fit(train_x, train_y)
            if adata.uns["_save_path_trained_models"]:
                pickle.dump(
                    rf,
                    open(
                        adata.uns["_save_path_trained_models"] + "rf_classifier.pkl",
                        "wb",
                    ),
                )
        else:
            rf = pickle.load(open(adata.uns["_save_path_trained_models"] + "rf_classifier.pkl", "rb"))
        adata.obs[self.result_key] = rf.predict(test_x)
        if adata.uns["_return_probabilities"]:
            adata.obs[self.result_key + "_probabilities"] = np.max(rf.predict_proba(test_x), axis=1)

    def compute_embedding(self, adata):
        pass

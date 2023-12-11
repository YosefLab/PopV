import logging
import pickle

import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV


class SVM:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layers_key: str | None = None,
        result_key: str | None = "popv_svm_prediction",
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
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def compute_integration(self, adata):
        pass

    def predict(self, adata):
        logging.info(f'Computing support vector machine. Storing prediction in adata.obs["{self.result_key}"]')
        test_x = adata.layers[self.layers_key] if self.layers_key else adata.X

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_x = adata[train_idx].layers[self.layers_key] if self.layers_key else adata[train_idx].X
            train_y = adata[train_idx].obs[self.labels_key].to_numpy()
            clf = CalibratedClassifierCV(svm.LinearSVC(**self.classifier_dict))
            clf.fit(train_x, train_y)
            if adata.uns["_save_path_trained_models"]:
                pickle.dump(
                    clf,
                    open(
                        adata.uns["_save_path_trained_models"] + "svm_classifier.pkl",
                        "wb",
                    ),
                )
        else:
            clf = pickle.load(open(adata.uns["_save_path_trained_models"] + "svm_classifier.pkl", "rb"))

        adata.obs[self.result_key] = clf.predict(test_x)

        if adata.uns["_return_probabilities"]:
            adata.obs[self.result_key + "_probabilities"] = np.max(clf.predict_proba(test_x), axis=1)

        adata.obs[self.result_key]

    def compute_embedding(self, adata):
        pass

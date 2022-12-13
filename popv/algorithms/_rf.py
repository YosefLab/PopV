import logging
from ast import Pass
from typing import Optional

from sklearn.ensemble import RandomForestClassifier


class RF:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        layers_key: Optional[str] = "logcounts",
        result_key: Optional[str] = "popv_rf_prediction",
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
        layers_key
            Key in layers field of adata used for classification.
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
        self.classifier_dict.update(classifier_dict)

    def compute_integration(self, adata):
        Pass

    def predict(self, adata):
        logging.info(
            'Computing random forest classifier. Storing prediction in adata.obs["{}"]'.format(
                self.result_key
            )
        )

        train_idx = adata.obs["_ref_subsample"]
        test_idx = adata.obs["_dataset"] == "query"

        train_x = adata[train_idx].layers[self.layers_key]
        train_y = adata[train_idx].obs[self.labels_key].to_numpy()
        test_x = adata[test_idx].layers[self.layers_key]

        rf = RandomForestClassifier(**self.classifier_dict)
        rf.fit(train_x, train_y)
        rf_pred = rf.predict(test_x)

        adata.obs[self.result_key] = adata.obs[self.labels_key]
        adata.obs.loc[test_idx, self.result_key] = rf_pred

    def compute_embedding(self, adata):
        pass

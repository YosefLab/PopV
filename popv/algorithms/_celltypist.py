import logging

import celltypist


class CELLTYPIST:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        result_key: str | None = "popv_celltypist_prediction",
        method_dict: dict | None = None,
        classifier_dict: dict | None = None,
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
        method_dict
            Additional parameters for celltypist training. Options at celltypist.train
        classifier_dict
            Dictionary to supply non-default values for celltypist annotation. Options at celltypist.annotate
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.result_key = result_key

        self.method_dict = {"check_expression": False, "n_jobs": 10, "max_iter": 500}
        if method_dict is not None:
            self.method_dict.update(method_dict)

        self.classifier_dict = {"mode": "best match", "majority_voting": True}
        if classifier_dict is not None:
            self.classifier_dict.update(classifier_dict)

    def compute_integration(self, adata):
        pass

    def predict(self, adata):
        logging.info(f'Saving celltypist results to adata.obs["{self.result_key}"]')

        if adata.uns["_prediction_mode"] == "retrain":
            train_idx = adata.obs["_ref_subsample"]
            train_adata = adata[train_idx].copy()
            model = celltypist.train(train_adata, self.labels_key, **self.method_dict)

            if adata.uns["_save_path_trained_models"]:
                model.write(adata.uns["_save_path_trained_models"] + "celltypist.pkl")
        if adata.uns["_prediction_mode"] == "fast":
            self.classifier_dict["majority_voting"] = False
        predictions = celltypist.annotate(
            adata,
            model=adata.uns["_save_path_trained_models"] + "celltypist.pkl",
            **self.classifier_dict,
        )
        out_column = (
            "majority_voting" if "majority_voting" in predictions.predicted_labels.columns else "predicted_labels"
        )

        adata.obs[self.result_key] = predictions.predicted_labels[out_column]
        if adata.uns["_return_probabilities"]:
            adata.obs[self.result_key + "_probabilities"] = predictions.probability_matrix.max(axis=1).values

    def compute_embedding(self, adata):
        pass

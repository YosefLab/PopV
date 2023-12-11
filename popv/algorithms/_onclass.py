import logging

import numpy as np
import obonet
import scipy
from OnClass.OnClassModel import OnClassModel


class ONCLASS:
    def __init__(
        self,
        batch_key: str | None = "_batch_annotation",
        labels_key: str | None = "_labels_annotation",
        layers_key: str | None = None,
        max_iter: int | None = 30,
        cell_ontology_obs_key: str | None = None,
        result_key: str | None = "popv_onclass_prediction",
        seen_result_key: str | None = "popv_onclass_seen",
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
            Layer in adata used for Onclass prediction.
        max_iter
            Maximum iteration in Onclass training
        cell_ontology_obs_key
            Key in obs in which ontology celltypes are stored.
        result_key
            Key in obs in which celltype annotation results are stored.
        """
        self.batch_key = batch_key
        self.labels_key = labels_key
        self.result_key = result_key
        self.seen_result_key = seen_result_key
        self.layers_key = layers_key

        if cell_ontology_obs_key is None:
            self.cell_ontology_obs_key = self.labels_key + "_cell_ontology_id"
        else:
            self.cell_ontology_obs_key = cell_ontology_obs_key
        self.shard_size = 50000
        self.max_iter = max_iter

    def make_celltype_to_cell_ontology_id_dict(self, cl_obo_file):
        """
        Make celltype to ontology id dict and vice versa.

        Parameters
        ----------
        cl_obo_file

        Returns
        -------
        name2id
            dictionary of celltype names to ontology id
        id2name
            dictionary of ontology id to celltype names
        """
        co = obonet.read_obo(cl_obo_file, encoding="utf-8")
        id2name = {id_: data.get("name") for id_, data in co.nodes(data=True)}
        id2name = {k: v for k, v in id2name.items() if v is not None}
        name2id = {v: k for k, v in id2name.items()}

        return name2id, id2name

    def make_cell_ontology_id(self, adata, celltype_dict, ontology_key):
        """
        Convert celltype names to ontology id.

        Parameters
        ----------
        adata
            AnnData object
        celltype_dict
            Dictionary mapping celltype to ontology id
        ontology_key
            Key in adata.obs to save ontology ids to.
            Default will be <labels_key>_cell_ontology_id
        """
        ontology_id = []
        single_warning = []  # print a single warning per missing cell-type

        for label in adata.obs[self.labels_key]:
            if label != adata.uns["unknown_celltype_label"]:
                if label not in celltype_dict:
                    if label not in single_warning:
                        logging.warning("Following label not in celltype_dict ", label)
                        single_warning.append(label)
                    ontology_id.append("unknown")
                ontology_id.append(celltype_dict[label])
            else:
                ontology_id.append("unknown")

        adata.obs[ontology_key] = ontology_id

    def compute_integration(self, adata):
        pass

    def predict(self, adata):
        logging.info(f'Computing Onclass. Storing prediction in adata.obs["{self.result_key}"]')
        adata.obs.loc[adata.obs["_dataset"] == "query", self.cell_ontology_obs_key] = adata.uns[
            "unknown_celltype_label"
        ]

        train_idx = adata.obs["_dataset"] == "ref"

        if self.layers_key is None:
            train_X = adata[train_idx].X.copy()
            test_X = adata.X.copy()
        else:
            train_X = adata[train_idx].layers[self.layers_key].copy()
            test_X = adata.layers[self.layers_key].copy()
        if scipy.sparse.issparse(train_X):
            train_X = train_X.todense()
            test_X = test_X.todense()

        cl_obo_file = adata.uns["_cl_obo_file"]
        cl_ontology_file = adata.uns["_cl_ontology_file"]
        nlp_emb_file = adata.uns["_nlp_emb_file"]

        celltype_dict, clid_2_name = self.make_celltype_to_cell_ontology_id_dict(cl_obo_file)
        self.make_cell_ontology_id(adata, celltype_dict, self.cell_ontology_obs_key)

        train_model = OnClassModel(cell_type_nlp_emb_file=nlp_emb_file, cell_type_network_file=cl_ontology_file)

        if adata.uns["_save_path_trained_models"] is not None:
            model_path = adata.uns["_save_path_trained_models"] + "/OnClass"
        else:
            model_path = None

        if adata.uns["_prediction_mode"] == "retrain":
            train_Y = adata[train_idx].obs[self.cell_ontology_obs_key]
            _ = train_model.EmbedCellTypes(train_Y)

            (
                corr_train_feature,
                corr_test_feature,
                corr_train_genes,
                _,
            ) = train_model.ProcessTrainFeature(
                train_X,
                train_Y,
                adata.var_names,
                test_feature=test_X,
                test_genes=adata.var_names,
                log_transform=False,
            )

            train_model.BuildModel(ngene=len(corr_train_genes))
            train_model.Train(
                corr_train_feature,
                train_Y,
                save_model=model_path,
                max_iter=self.max_iter,
            )
        else:
            train_model.BuildModel(ngene=None, use_pretrain=model_path)

        adata.obs[self.result_key] = None

        corr_test_feature = train_model.ProcessTestFeature(
            test_feature=test_X,
            test_genes=adata.var_names,
            use_pretrain=model_path,
            log_transform=False,
        )

        if adata.uns["_prediction_mode"] == "fast":
            onclass_seen = np.argmax(train_model.model.predict(corr_test_feature), axis=1)
            pred_label = [train_model.i2co[ind] for ind in onclass_seen]
            pred_label_str = [clid_2_name[ind] for ind in pred_label]
            adata.obs[self.result_key] = pred_label_str
            adata.obs["onclass_seen"] = pred_label_str
        else:
            onclass_pred = train_model.Predict(corr_test_feature, use_normalize=False, refine=True, unseen_ratio=-1.0)
            pred_label = [train_model.i2co[ind] for ind in onclass_pred[2]]
            pred_label_str = [clid_2_name[ind] for ind in pred_label]
            adata.obs[self.result_key] = pred_label_str

            onclass_seen = np.argmax(onclass_pred[0], axis=1)
            pred_label = [train_model.i2co[ind] for ind in onclass_seen]
            pred_label_str = [clid_2_name[ind] for ind in pred_label]
            adata.obs[self.seen_result_key] = pred_label_str

            if adata.uns["_return_probabilities"]:
                adata.obs[self.result_key + "_probabilities"] = np.max(onclass_pred[1], axis=1) / onclass_pred[1].sum(1)
                adata.obsm["onclass_probabilities"] = onclass_pred[1] / onclass_pred[1].sum(1, keepdims=True)
                adata.obs["popv_onclass_seen" + "_probabilities"] = np.max(onclass_pred[0], axis=1)

    def compute_embedding(self, adata):
        return None

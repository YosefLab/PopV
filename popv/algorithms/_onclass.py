import logging
from typing import Optional

import obonet
from OnClass.OnClassModel import OnClassModel


class ONCLASS:
    def __init__(
        self,
        batch_key: Optional[str] = "_batch_annotation",
        labels_key: Optional[str] = "_labels_annotation",
        layers_key: Optional[str] = None,
        max_iter: Optional[int] = 20,
        cell_ontology_obs_key: Optional[str] = None,
        result_key: Optional[str] = "popv_onclass_prediction",
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
        self.layers_key = layers_key

        if cell_ontology_obs_key is None:
            self.cell_ontology_obs_key = self.labels_key + "_cell_ontology_id"
        else:
            self.cell_ontology_obs_key = cell_ontology_obs_key
        self.shard_size = 10000
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
        with open(cl_obo_file) as f:
            co = obonet.read_obo(f)
            id2name = {id_: data.get("name") for id_, data in co.nodes(data=True)}
            id2name = {k: v.lower() for k, v in id2name.items() if v is not None}
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

        for label in adata.obs[self.labels_key]:
            if label != adata.uns["unknown_celltype_label"]:
                if label not in celltype_dict:
                    print("Following label not in celltype_dict ", label)
                ontology_id.append(celltype_dict[label])
            else:
                ontology_id.append("unknown")

        adata.obs[ontology_key] = ontology_id

    def compute_integration(self, adata):
        pass

    def predict(self, adata):
        logging.info(
            'Computing Onclass. Storing prediction in adata.obs["{}"]'.format(
                self.result_key
            )
        )
        adata.obs.loc[
            adata.obs["_dataset"] == "query", self.cell_ontology_obs_key
        ] = adata.uns["unknown_celltype_label"]

        cl_obo_file = adata.uns["_cl_obo_file"]
        cl_ontology_file = adata.uns["_cl_ontology_file"]
        nlp_emb_file = adata.uns["_nlp_emb_file"]

        celltype_dict, clid_2_name = self.make_celltype_to_cell_ontology_id_dict(
            cl_obo_file
        )
        self.make_cell_ontology_id(adata, celltype_dict, self.cell_ontology_obs_key)

        train_model = OnClassModel(
            cell_type_nlp_emb_file=nlp_emb_file, cell_type_network_file=cl_ontology_file
        )

        train_idx = adata.obs["_dataset"] == "ref"
        test_idx = adata.obs["_dataset"] == "query"

        if self.layers_key is None:
            train_X = adata[train_idx].X.todense()
            test_X = adata[test_idx].X.todense()
        else:
            train_X = adata[train_idx].layers[self.layers_key].todense()
            test_X = adata[test_idx].layers[self.layers_key].todense()

        train_Y = adata[train_idx].obs[self.cell_ontology_obs_key]

        test_adata = adata[test_idx]

        _ = train_model.EmbedCellTypes(train_Y)
        model_path = "OnClass"

        (
            corr_train_feature,
            corr_test_feature,
            corr_train_genes,
            corr_test_genes,
        ) = train_model.ProcessTrainFeature(
            train_X,
            train_Y,
            adata.var_names,
            test_feature=test_X,
            test_genes=adata.var_names,
        )
        train_model.BuildModel(ngene=len(corr_train_genes))

        train_model.Train(
            corr_train_feature, train_Y, save_model=model_path, max_iter=self.max_iter
        )

        test_adata.obs[self.result_key] = None

        corr_test_feature = train_model.ProcessTestFeature(
            corr_test_feature,
            corr_test_genes,
            use_pretrain=model_path,
            log_transform=False,
        )

        if test_adata.n_obs > self.shard_size:
            for i in range(0, test_adata.n_obs, self.shard_size):
                tmp_X = corr_test_feature[i : i + self.shard_size]
                onclass_pred = train_model.Predict(tmp_X, use_normalize=False)
                pred_label_str = [train_model.i2co[l] for l in onclass_pred[2]]
                pred_label_str = [clid_2_name[i] for i in pred_label_str]
                test_adata.obs[self.result_key][
                    i : i + self.shard_size
                ] = pred_label_str
        else:
            onclass_pred = train_model.Predict(corr_test_feature, use_normalize=False)
            pred_label_str = [train_model.i2co[l] for l in onclass_pred[2]]
            pred_label_str = [clid_2_name[i] for i in pred_label_str]
            test_adata.obs[self.result_key] = pred_label_str

        adata.obs[self.result_key] = adata.obs[self.labels_key].astype(str)
        adata.obs.loc[test_adata.obs_names, self.result_key] = test_adata.obs[
            self.result_key
        ]

    def compute_embedding(self, adata):
        pass

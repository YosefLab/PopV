import os
import obonet

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp_sparse
import scanorama
import scanpy as sc
import scvi
import seaborn as sns
import string



from OnClass.OnClassModel import OnClassModel
import logging

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.backends.backend_pdf
from numba import boolean, float32, float64, int32, int64, vectorize
from collections import defaultdict

from .utils import *


@try_method("Integrating data with bbknn")
def run_bbknn(adata, batch_key="_batch"):
    sc.external.pp.bbknn(
        adata,
        batch_key=batch_key,
        approx=True,
        metric="angular",
        n_pcs=20,
        trim=None,
        annoy_n_trees=10,
        use_faiss=True,
        set_op_mix_ratio=1.0,
        local_connectivity=1,
    )
    adata.obsm["bbknn_umap"] = sc.tl.umap(adata, maxiter=1500, copy=True).obsm['X_umap']
    return adata


@try_method("Classifying with knn on bbknn distances")
def run_knn_on_bbknn(
    adata,
    labels_key="_labels_annotation",
    result_key="knn_on_bbknn_pred",
):
    distances = adata.obsp["distances"]

    ref_idx = adata.obs["_dataset"] == "ref"
    query_idx = adata.obs["_dataset"] == "query"

    ref_dist_idx = np.where(ref_idx == True)[0]
    query_dist_idx = np.where(query_idx == True)[0]

    train_y = adata[ref_idx].obs[labels_key].to_numpy()
    train_distances = distances[ref_dist_idx, :][:, ref_dist_idx]

    knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed", weights='uniform')
    knn.fit(train_distances, y=train_y)

    test_distances = distances[query_dist_idx, :][:, ref_dist_idx]
    knn_pred = knn.predict(test_distances)

    # save_results. ref cells get ref annotations, query cells get predicted
    adata.obs[result_key] = adata.obs[labels_key]
    adata.obs[result_key][query_idx] = knn_pred
    print('Saved knn on bbknn results to adata.obs["{}"]'.format(result_key))


@try_method("Classifying with random forest")
def run_rf_on_hvg(
    adata,
    labels_key="_labels_annotation",
    save_key="rf_pred",
    layers_key="logcounts",
):
    train_idx = adata.obs["_ref_subsample"]
    test_idx = adata.obs["_dataset"] == "query"

    train_x = adata[train_idx].layers[layers_key]
    train_y = adata[train_idx].obs[labels_key].to_numpy()
    test_x = adata[test_idx].layers[layers_key]
    
    print("Training random forest classifier with {} cells".format(len(train_y)))
    n_features = np.max([200., np.sqrt(2000.)]).astype(int)
    rf = RandomForestClassifier(class_weight='balanced_subsample', max_features=n_features)
    rf.fit(train_x, train_y)
    rf_pred = rf.predict(test_x)

    adata.obs[save_key] = adata.obs[labels_key]
    adata.obs[save_key][test_idx] = rf_pred

@try_method("Classifying with onclass")
def run_onclass(
    adata,
    cl_obo_file,
    cl_ontology_file,
    nlp_emb_file,
    labels_key="_labels_annotation",
    layer=None,
    save_key="onclass_pred",
    n_hidden=500,
    max_iter=20,
    save_model="onclass_model",
    shard_size=50000
):
    celltype_dict, clid_2_name = make_celltype_to_cell_ontology_id_dict(cl_obo_file)
    cell_ontology_obs_key = make_cell_ontology_id(adata, labels_key, celltype_dict)

    train_model = OnClassModel(
        cell_type_nlp_emb_file=nlp_emb_file, cell_type_network_file=cl_ontology_file
    )

    train_idx = adata.obs["_dataset"] == "ref"
    test_idx = adata.obs["_dataset"] == "query"

    if layer is None:
        train_X = adata[train_idx].X.todense()
        test_X = adata[test_idx].X.todense()
    else:
        train_X = adata[train_idx].layers[layer].todense()
        test_X = adata[test_idx].layers[layer].todense()

    train_genes = adata[train_idx].var_names
    test_genes = adata[test_idx].var_names
    train_Y = adata[train_idx].obs[cell_ontology_obs_key]

    test_adata = adata[test_idx]

    _ = train_model.EmbedCellTypes(train_Y)
    model_path = "OnClass"
    

    corr_train_feature, corr_test_feature, corr_train_genes, corr_test_genes = train_model.ProcessTrainFeature(train_X, train_Y, train_genes, test_feature=test_X, test_genes=test_genes)
    train_model.BuildModel(ngene=len(corr_train_genes))
    
    train_model.Train(
        corr_train_feature, train_Y, save_model=model_path, max_iter=max_iter
    )

    test_adata.obs[save_key] = "na"
    
    corr_test_feature = train_model.ProcessTestFeature(corr_test_feature, corr_test_genes, use_pretrain = model_path, log_transform = False)

    if test_adata.n_obs > shard_size:
        for i in range(0, test_adata.n_obs, shard_size):
            tmp_X = corr_test_feature[i : i + shard_size]
            onclass_pred = train_model.Predict(tmp_X, use_normalize=False)
            pred_label_str = [train_model.i2co[l] for l in onclass_pred[2]]
            pred_label_str = [clid_2_name[i] for i in pred_label_str]
            test_adata.obs[save_key][i : i + shard_size] = pred_label_str
    else:
        onclass_pred = train_model.Predict(corr_test_feature, use_normalize=False)
        pred_label_str = [train_model.i2co[l] for l in onclass_pred[2]]
        pred_label_str = [clid_2_name[i] for i in pred_label_str]
        test_adata.obs[save_key] = pred_label_str

    adata.obs[save_key] = adata.obs[labels_key].astype(str)
    adata.obs[save_key][test_adata.obs_names] = test_adata.obs[save_key]

    return adata

@try_method("Classifying with SVM")
def run_svm_on_hvg(
    adata,
    labels_key="_labels_annotation",
    save_key="svm_pred",
    layers_key="logcounts",
):
    train_idx = adata.obs["_ref_subsample"]
    test_idx = adata.obs["_dataset"] == "query"

    train_x = adata[train_idx].layers[layers_key]
    train_y = adata[train_idx].obs[labels_key].to_numpy()
    test_x = adata[test_idx].layers[layers_key]

    clf = svm.LinearSVC(max_iter=5000, class_weight='balanced')
    clf.fit(train_x, train_y)
    svm_pred = clf.predict(test_x, )

    # save_results
    adata.obs[save_key] = adata.obs[labels_key]
    adata.obs[save_key][test_idx] = svm_pred


@try_method("Running scvi")
def run_scvi(
    adata,
    n_latent=50,
    n_layers=2,
    dropout_rate=0.1,
    dispersion="gene-batch",
    max_epochs=None,
    batch_size=256,
    pretrained_scvi_path=None,
    var_subset_type="inner_join",
    obsm_latent_key="X_scvi",
    save_folder=None,
    overwrite=True,
    save_anndata=False,
):
    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="_batch_annotation",
        labels_key="_labels_annotation",
        layer="scvi_counts",
    ) 
    training_mode = adata.uns["_training_mode"]
    if training_mode == "online" and pretrained_scvi_path is None:
        raise ValueError("online training but no pretrained_scvi_path passed in.")

    if training_mode == "offline":
        model = scvi.model.SCVI(
            adata,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            use_layer_norm="both",
            use_batch_norm="none",
            encode_covariates=True,
        )
        print("Training scvi offline.")

    elif training_mode == "online":
        if max_epochs is None:
            n_cells = adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 200), 200])

        query = adata[adata.obs["_dataset"] == "query"].copy()
        model = scvi.model.SCVI.load_query_data(query, pretrained_scvi_path)
        print("Training scvi online.")

    model.train(max_epochs=max_epochs, train_size=1.0, batch_size=batch_size)

    # # temporary scvi hack
    # tmp_mappings = adata.uns["_scvi"]["categorical_mappings"]["_scvi_labels"]
    # model.scvi_setup_dict_["categorical_mappings"]["_scvi_labels"] = tmp_mappings

    adata.obsm[obsm_latent_key] = model.get_latent_representation(adata)

    sc.pp.neighbors(adata, use_rep=obsm_latent_key)
    adata.obsm[obsm_latent_key + "_umap"] = sc.tl.umap(adata, min_dist=0.01, maxiter=1500, copy=True).obsm['X_umap']

    if save_folder is not None:
        print ("Saving scvi model to ", save_folder)
        model.save(save_folder, overwrite=overwrite, save_anndata=save_anndata)

@try_method("Classifying with knn on scVI latent space")
def run_knn_on_scvi(
    adata,
    labels_key="_labels_annotation",
    obsm_key="X_scvi",
    result_key="knn_on_scvi_pred",
):
    if obsm_key not in adata.obsm.keys():
        raise ValueError("Please train scVI first or pass in a valid obsm_key.")

    print(
        "Training knn on scvi latent space. "
        + 'Using latent space in adata.obsm["{}"]'.format(obsm_key)
    )

    ref_idx = adata.obs["_dataset"] == "ref"
    query_idx = adata.obs["_dataset"] == "query"

    train_X = adata[ref_idx].obsm[obsm_key]
    train_Y = adata[ref_idx].obs[labels_key].to_numpy()

    test_X = adata[query_idx].obsm[obsm_key]

    knn = KNeighborsClassifier(n_neighbors=15, weights="uniform")
    knn.fit(train_X, train_Y)
    knn_pred = knn.predict(test_X)

    # save_results
    adata.obs[result_key] = adata.obs[labels_key]
    adata.obs[result_key][query_idx] = knn_pred


@try_method("Classifying with knn on scanorama latent space")
def run_knn_on_scanorama(
    adata,
    labels_key="_labels_annotation",
    obsm_key="X_scanorama",
    result_key="knn_on_scanorama_pred",
):
    print("Running knn on scanorama")
    if obsm_key not in adata.obsm.keys():
        print("Please run scanorama first or pass in a valid obsm_key.")

    ref_idx = adata.obs["_dataset"] == "ref"
    query_idx = adata.obs["_dataset"] == "query"

    train_X = adata[ref_idx].obsm[obsm_key]
    train_Y = adata[ref_idx].obs["_labels_annotation"].to_numpy()

    test_X = adata[query_idx].obsm[obsm_key]

    knn = KNeighborsClassifier(n_neighbors=15, weights="uniform")
    knn.fit(train_X, train_Y)
    knn_pred = knn.predict(test_X)

    # save_results
    adata.obs[result_key] = adata.obs[labels_key]
    adata.obs[result_key][query_idx] = knn_pred


@try_method("Running scanorama")
def run_scanorama(adata, batch_key="_batch"):
    # TODO add check if in colab and n_genes > 120000
    # throw warning
    adatas = [adata[adata.obs[batch_key] == i] for i in np.unique(adata.obs[batch_key])]
    scanorama.integrate_scanpy(adatas, dimred=50)
    tmp_adata = anndata.concat(adatas)
    adata.obsm["X_scanorama"] = tmp_adata[adata.obs_names].obsm["X_scanorama"]

    print("Computing umap on scanorama")
    sc.pp.neighbors(adata, use_rep="X_scanorama")
    sc.tl.umap(adata, maxiter=1500)
    adata.obsm["scanorama_umap"] = sc.tl.umap(adata, min_dist=0.01, maxiter=1500, copy=True).obsm['X_umap']


@try_method("Running scANVI")
def run_scanvi(
    adata,
    unlabeled_category="unknown",
    n_layers=2,
    dropout_rate=0.1,
    n_classifier_layers=1,
    classifier_dropout=0.4,
    max_epochs=None,
    n_latent=50,
    batch_size=256,
    dispersion='gene-batch',
    n_epochs_kl_warmup=20,
    n_samples_per_label=100,
    obsm_latent_key="X_scanvi",
    obs_pred_key="scanvi_pred",
    pretrained_scanvi_path=None,
    save_folder=None,
    save_anndata=False,
    overwrite=True,
):
    scvi.model.SCANVI.setup_anndata(
        adata,
        batch_key="_batch_annotation",
        labels_key="_labels_annotation",
        layer="scvi_counts",
        unlabeled_category = unlabeled_category
    ) 
    training_mode = adata.uns["_training_mode"]
    if training_mode == "online" and pretrained_scanvi_path is None:
        raise ValueError("online training but no pretrained_scvi_path passed in.")

    if training_mode == "offline":
        model_kwargs = dict(
            dispersion=dispersion,
            use_layer_norm="both",
            use_batch_norm="none",
            classifier_parameters={
                "n_layers": n_classifier_layers,
                "dropout_rate": classifier_dropout,
            },
        )
        model = scvi.model.SCANVI(
            adata,
            n_layers=n_layers,
            encode_covariates=True,
            dropout_rate=dropout_rate,
            n_latent=n_latent,
            **model_kwargs,
        )

    elif training_mode == "online":
        if max_epochs is None:
            n_cells = adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 200), 200])

        query = adata[adata.obs["_dataset"] == "query"].copy()
        model = scvi.model.SCANVI.load_query_data(
            query, pretrained_scanvi_path, freeze_classifier=True
        )

    plan_kwargs = dict(n_epochs_kl_warmup=n_epochs_kl_warmup)
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        train_size=1.0,
        n_samples_per_label=n_samples_per_label,
        plan_kwargs=plan_kwargs,
    )

    adata.obsm[obsm_latent_key] = model.get_latent_representation(adata)
    adata.obs[obs_pred_key] = model.predict(adata)

    if save_folder is not None:
        print ("Saving scanvi model to ", save_folder)
        model.save(save_folder, overwrite=overwrite, save_anndata=save_anndata)


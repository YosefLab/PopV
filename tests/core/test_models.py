"""Test various algorithms implemented in PopV."""
import os
from os.path import exists

import anndata
import numpy as np
import scanpy as sc

import popv
from popv.preprocessing import Process_Query


def _get_test_anndata():
    print(os.getcwd())
    save_folder = "popv_test_results/"
    fn = save_folder + "annotated_query.h5ad"
    if exists(save_folder + fn):
        return anndata.read(save_folder + fn)

    ref_adata_path = "dataset/test/ts_lung_subset.h5ad"
    ref_adata = sc.read(ref_adata_path)

    query_adata_path = "dataset/test/lca_subset.h5ad"
    query_adata = sc.read(query_adata_path)
    assert query_adata.n_vars == query_adata.X.shape[1]

    ref_labels_key = "cell_ontology_class"
    ref_batch_key = "donor_assay"
    min_celltype_size = np.min(ref_adata.obs.groupby("cell_ontology_class").size())
    n_samples_per_label = np.max((min_celltype_size, 20))

    query_batch_key = None

    # Lesser used parameters
    query_labels_key = None
    unknown_celltype_label = "unknown"
    adata = Process_Query(
        query_adata,
        ref_adata,
        query_batch_key=query_batch_key,
        query_labels_key=query_labels_key,
        ref_labels_key=ref_labels_key,
        ref_batch_key=ref_batch_key,
        unknown_celltype_label=unknown_celltype_label,
        save_path_trained_models=save_folder,
        prediction_mode="retrain",
        n_samples_per_label=n_samples_per_label,
        compute_embedding=True,
        return_probabilities=True,
        use_gpu=False,
        hvg=4000,
    )

    return adata


def test_bbknn():
    """Test BBKNN algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.knn_on_bbknn()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_knn_on_bbknn_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_on_bbknn_prediction"].isnull().any()


def test_onclass():
    """Test Onclass algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.onclass(
        max_iter=2,
    )
    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_onclass_prediction" in adata.obs.columns
    assert not adata.obs["popv_onclass_prediction"].isnull().any()


def test_rf():
    """Test Random Forest algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.rf()
    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_rf_prediction" in adata.obs.columns
    assert not adata.obs["popv_rf_prediction"].isnull().any()


def test_scanorama():
    """Test Scanorama algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.knn_on_scanorama()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_knn_on_scanorama_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_on_scanorama_prediction"].isnull().any()


def test_harmony():
    """Test Harmony algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.knn_on_harmony()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_knn_on_harmony_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_on_harmony_prediction"].isnull().any()


def test_scanvi():
    """Test SCANVI algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.scanvi(
        n_epochs_unsupervised=5,
    )

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_scanvi_prediction" in adata.obs.columns
    assert not adata.obs["popv_scanvi_prediction"].isnull().any()


def test_scvi():
    """Test SCVI algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.knn_on_scvi(max_epochs=3)

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_knn_on_scvi_prediction" in adata.obs.columns
    assert not adata.obs["popv_knn_on_scvi_prediction"].isnull().any()


def test_svm():
    """Test Support Vector Machine algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.svm()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_svm_prediction" in adata.obs.columns
    assert not adata.obs["popv_svm_prediction"].isnull().any()


def test_celltypist():
    """Test Celltypist algorithm."""
    adata = _get_test_anndata().adata
    current_method = popv.algorithms.celltypist()

    current_method.compute_integration(adata)
    current_method.predict(adata)
    current_method.compute_embedding(adata)

    assert "popv_celltypist_prediction" in adata.obs.columns
    assert not adata.obs["popv_celltypist_prediction"].isnull().any()


def test_annotation():
    """Test Annotation and Plotting pipeline."""
    adata = _get_test_anndata().adata
    popv.annotation.annotate_data(adata, methods=["svm", "rf"], save_path=None)
    popv.visualization.agreement_score_bar_plot(adata)
    popv.visualization.prediction_score_bar_plot(adata)
    popv.visualization.make_agreement_plots(
        adata, prediction_keys=adata.uns["prediction_keys"]
    )
    popv.visualization.celltype_ratio_bar_plot(adata)

    assert "popv_majority_vote_prediction" in adata.obs.columns
    assert not adata.obs["popv_majority_vote_prediction"].isnull().any()


if __name__ == "__main__":
    test_bbknn()
    test_onclass()
    test_rf()
    test_scanorama()
    test_scanvi()
    test_scvi()
    test_svm()
    test_annotation()

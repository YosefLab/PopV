from popv import *
import anndata
import popv
import numpy as np
from os.path import exists
import scanpy as sc
import scvi
from scvi.data import synthetic_iid
from IPython.core.debugger import set_trace


def get_test_anndata():
    save_folder = 'popv_test_results/'
    fn = save_folder + "annotated_query.h5ad"
    if exists(save_folder + fn):
        return anndata.read(save_folder + fn)
    

    
    ref_adata_path = '/home/galen/ts_annotation/data/paper/lung_eval/TSP1_TSP15_lung.h5ad'
    ref_adata = anndata.read(ref_adata_path)
    
    query_adata = synthetic_iid(n_genes = ref_adata.n_vars)
    query_adata.var.index = ref_adata.var.index.copy()
    ref_labels_key='cell_ontology_class'
    ref_batch_key = ['donor', 'method']
    min_celltype_size = np.min(ref_adata.obs.groupby('cell_ontology_class').size())
    n_samples_per_label = np.max((min_celltype_size, 100))

    query_batch_key = None
    methods = ['bbknn','scvi', 'scanvi', 'svm', 'rf', 'onclass', 'scanorama']
    training_mode='offline'

    # Lesser used parameters
    query_labels_key=None
    unknown_celltype_label='unknown'
    adata = process_query(query_adata,
                      ref_adata,
                      save_folder=save_folder,
                      query_batch_key=query_batch_key,
                      query_labels_key=query_labels_key,
                      unknown_celltype_label=unknown_celltype_label,
                      pretrained_scvi_path=None,
                      ref_labels_key=ref_labels_key, 
                      ref_batch_key=ref_batch_key,
                      training_mode=training_mode,
                      ref_adata_path=ref_adata_path,
                      n_samples_per_label=n_samples_per_label)
    
    return adata

def test_onclass():
    onclass_ontology_file="../ontology/cl.ontology"
    onclass_obo_fp="../ontology/cl.obo"
    onclass_emb_fp="../ontology/cl.ontology.nlp.emb"
    adata = get_test_anndata()
    adata = run_onclass(
        adata=adata,
        layer="scvi_counts",
        max_iter=2,
        cl_obo_file = onclass_obo_fp,
        cl_ontology_file=onclass_ontology_file,
        nlp_emb_file=onclass_emb_fp
    )
    
def test_scanvi():
    training_mode = 'offline'
    obsm_latent_key = "X_scanvi_{}".format(training_mode)
    predictions_key = "scanvi_{}_pred".format(training_mode)
    adata = get_test_anndata()
    run_scanvi(
        adata,
        max_epochs=None,
        n_latent=100,
        dropout_rate=0.1,
        obsm_latent_key=obsm_latent_key,
        obs_pred_key=predictions_key,
        pretrained_scanvi_path=None,
    )
if __name__ == "__main__":
    test_onclass()
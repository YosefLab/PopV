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

def make_cell_ontology_id(adata, labels_key, celltype_dict, ontology_key=None):
    """
    Convert celltype names to ontology id.

    Parameters
    ----------
    adata
        AnnData object
    labels_key
        Key in adata.obs to convert to ontology id
    celltype_dict
        Dictionary mapping celltype to ontology id
    ontology_key
        Key in adata.obs to save ontology ids to.
        Default will be <labels_key>_cell_ontology_id
    """
    if ontology_key is None:
        ontology_key = labels_key + "_cell_ontology_id"
    ontology_id = []

    for label in adata.obs[labels_key]:
        if label != "unknown":
            if label not in celltype_dict:
                print("Following label not in celltype_dict ", label)
            ontology_id.append(celltype_dict[label])
        else:
            ontology_id.append("unknown")

    adata.obs[ontology_key] = ontology_id
    return ontology_key


def make_celltype_to_cell_ontology_id_dict(obo_file):
    """
    Make celltype to ontology id dict and vice versa.

    Parameters
    ----------
    obo_file
        obofile to read

    Returns
    -------
    name2id
        dictionary of celltype names to ontology id
    id2name
        dictionary of ontology id to celltype names
    """
    with open(obo_file, "r") as f:
        co = obonet.read_obo(f)
        id2name = {id_: data.get("name") for id_, data in co.nodes(data=True)}
        id2name = {k: v.lower() for k, v in id2name.items() if v is not None}
        name2id = {v: k for k, v in id2name.items()}

    return name2id, id2name

def get_pretrained_model_genes(scvi_model_path):
    """
    Get the genes used to train a saved scVI model

    Parameters
    ----------
    scvi_model_path
        Path to saved scvi model

    Returns
    -------
    var_names
        Names of genes used to train the saved scvi model
    """
    varnames_path = os.path.join(scvi_model_path, "var_names.csv")
    var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)
    return var_names
def try_method(log_message):
    """
    Decorator which will except an Exception if it failed.
    """

    def try_except(func):
        def wrapper(*args, **kwargs):
            try:
                print("{}.".format(log_message))
                func(*args, **kwargs)
            except Exception as e:
                print("{} failed. Skipping.".format(log_message))
                print(e)

        return wrapper

    return try_except

def subsample_dataset(
    adata,
    labels_key,
    n_samples_per_label=100,
    ignore_label=None,
):
    """
    Subsamples dataset per label to n_samples_per_label.

    If a label has fewer than n_samples_per_label examples, then will use
    all the examples. For labels in ignore_label, they won't be included
    in the resulting subsampled dataset.

    Parameters
    ----------
    adata
        AnnData object
    labels_key
        Key in adata.obs for label information
    n_samples_per_label
        Maximum number of samples to use per label
    ignore_label
        List of labels to ignore (not subsample).

    Returns
    -------
    Returns list of obs_names corresponding to subsampled dataset

    """
    sample_idx = []
    labels, counts = np.unique(adata.obs[labels_key], return_counts=True)

    print("Sampling {} per label".format(n_samples_per_label))

    for label in ignore_label:
        if label in labels:
            idx = np.where(labels == label)
            labels = np.delete(labels, idx)
            counts = np.delete(counts, idx)

    for i, label in enumerate(labels):
        label_locs = np.where(adata.obs[labels_key] == label)[0]
        if counts[i] < n_samples_per_label:
            sample_idx.append(label_locs)
        else:
            label_subset = np.random.choice(
                label_locs, n_samples_per_label, replace=False
            )
            sample_idx.append(label_subset)
    sample_idx = np.concatenate(sample_idx)
    return adata.obs_names[sample_idx]

def check_genes_is_subset(ref_genes, query_genes):
    """
    Check whether query_genes is a subset of ref_genes.

    Parameters
    ----------
    ref_genes
        List of reference genes
    query_genes
        List of query genes

    Returns
    -------
    is_subset
        True if it is a subset, False otherwise.

    """
    if len(set(query_genes)) != len(query_genes):
        print("Warning: Your genes are not unique.")

    if set(ref_genes).issubset(set(query_genes)):
        print("All ref genes are in query dataset. Can use pretrained models.")
        is_subset = True
    else:
        print("Not all reference genes are in query dataset. Retraining models.")
        is_subset = False
    return is_subset



def make_batch_covariate(adata, batch_keys, new_batch_key):
    """
    Combines all the batches in batch_keys into a single batch.
    Saves results into adata.obs['_batch']

    Parameters
    ----------
    adata
        Anndata object
    batch_keys
        List of keys in adat.obs corresponding to batches
    """
    adata.obs[new_batch_key] = ""
    for key in batch_keys:
        v1 = adata.obs[new_batch_key].values
        v2 = adata.obs[key].values
        adata.obs[new_batch_key] = [a + b for a, b in zip(v1, v2)]

@vectorize(
    [
        boolean(int32),
        boolean(int64),
        boolean(float32),
        boolean(float64),
    ],
    target="parallel",
    cache=True,
)
def _is_not_count(d):
    return d < 0 or d % 1 != 0


def _check_nonnegative_integers(data):
    """Approximately checks values of data to ensure it is count data."""
    if isinstance(data, np.ndarray):
        data = data
    elif issubclass(type(data), sp_sparse.spmatrix):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    else:
        raise TypeError("data type not understood")

    n = len(data)
    inds = np.random.permutation(n)[:20]
    check = data.flat[inds]
    return ~np.any(_is_not_count(check))

def calculate_depths(g):
    depths = {}
    
    for node in g.nodes():
        path = nx.shortest_path_length(g,node)
        if "cell" not in path:
            print('Celltype not in DAG: ', node)
        else:
            depth = path['cell']
        depths[node] = depth
    
    return depths



def save_results(
    adata, results_adata_path, obs_keys=[], obsm_keys=[], compression="gzip"
):
    """
    If results_adata_path exists, will load and save results into it
    Else, will save adata to results_adata_path

    Parameters
    ----------
    adata
          adata with results in it
    results_adata_path
          path to save results. If it already exists, will load and save data to it
    obs_keys
          obs keys to save
    obsm_keys
          obsm keys to save
    compression
          If enabled, will save with compression. Smaller file sizes, but longer save times
    """
    if os.path.exists(results_adata_path):
        results = anndata.read(results_adata_path)
        for key in obs_keys:
            if key in adata.obs.keys():
                results.obs[key] = 'na'
                # results.obs[key] = adata[results.obs_names].obs[key]
                results.obs[key].update(results.obs[key])
        for key in obsm_keys:
            if key in adata.obsm.keys():
                results.obsm[key] = 'na'
                # results.obsm[key] = adata[results.obs_names].obsm[key]
                results.obsm[key].update(results.obsm[key])
        results.write(results_adata_path, compression)
    else:
        adata.write(results_adata_path, compression)
        
        
        
def make_ontology_dag(obofile, lowercase=True):
    """
    Returns ontology DAG from obofile
    """
    co = obonet.read_obo(obofile)
    id_to_name = {id_: data.get('name') for id_, data in co.nodes(data=True)}
    name_to_id = {data['name']: id_ for id_, data in co.nodes(data=True) if ('name' in data)}
    
    #get all node ids that are celltypes (start with CL)
    cl_ids = {id_:True for name, id_ in name_to_id.items() if id_.startswith('CL:')}
    
    # make new empty graph
    g = nx.MultiDiGraph()

    # add all celltype nodes to graph
    for node in co.nodes():
        if node in cl_ids:
            nodename = id_to_name[node]
            g.add_node(nodename)
    
    # iterate over 
    for node in co.nodes():
        if node in cl_ids:
            for child, parent, key in co.out_edges(node, keys=True):
                if child.startswith('CL:') and parent.startswith('CL:') and key=='is_a':
                    childname = id_to_name[child]
                    parentname = id_to_name[parent]
                    g.add_edge(childname, parentname, key=key)
    
    assert nx.is_directed_acyclic_graph(g) is True
    
    if lowercase:
        mapping = {s:s.lower() for s in list(g.nodes)}
        g = nx.relabel_nodes(g, mapping)
    return g
import logging

import networkx as nx
import numpy as np
import obonet


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
    labels_counts = dict(adata.obs[labels_key].value_counts())

    logging.info(f"Sampling {n_samples_per_label} per label")

    for label in ignore_label:
        labels_counts.pop(label, None)

    for label in labels_counts.keys():
        label_locs = np.where(adata.obs[labels_key] == label)[0]
        if labels_counts[label] < n_samples_per_label:
            sample_idx.append(label_locs)
        else:
            label_subset = np.random.choice(label_locs, n_samples_per_label, replace=False)
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
        logging.warning("Genes in query_dataset are not unique.")

    if set(ref_genes).issubset(set(query_genes)):
        logging.info("All ref genes are in query dataset. Can use pretrained models.")
        is_subset = True
    else:
        logging.info("Not all reference genes are in query dataset. Set 'prediction_mode' to 'retrain'.")
        is_subset = False
    return is_subset


def make_batch_covariate(adata, batch_keys, new_batch_key):
    """
    Combine all the batches in batch_keys into a single batch. Save result into adata.obs['_batch']

    Parameters
    ----------
    adata
        Anndata object
    batch_keys
        List of keys in adat.obs corresponding to batches
    """
    adata.obs[new_batch_key] = adata.obs[batch_keys].astype(str).sum(1).astype("category")


def calculate_depths(g):
    """
    Calculate depth of each node in a network.

    Parameters
    ----------
    g
        Graph object to compute path_length.

    Returns
    -------
    depths
        Dictionary containing depth for each node

    """
    depths = {}

    for node in g.nodes():
        path = nx.shortest_path_length(g, node)
        if "cell" not in path:
            logging.warning("Celltype not in DAG: ", node)
        else:
            depth = path["cell"]
        depths[node] = depth

    return depths


def make_ontology_dag(obofile, lowercase=False):
    """
    Construct a graph with all cell-types.

    Parameters
    ----------
    obofile
        File with all ontology cell-types.

    Returns
    -------
    g
        Graph containing all cell-types
    """
    co = obonet.read_obo(obofile, encoding="utf-8")
    id_to_name = {id_: data.get("name") for id_, data in co.nodes(data=True)}
    name_to_id = {data["name"]: id_ for id_, data in co.nodes(data=True) if ("name" in data)}

    # get all node ids that are celltypes (start with CL)
    cl_ids = {id_: True for _, id_ in name_to_id.items() if id_.startswith("CL:")}

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
                if child.startswith("CL:") and parent.startswith("CL:") and key == "is_a":
                    childname = id_to_name[child]
                    parentname = id_to_name[parent]
                    g.add_edge(childname, parentname, key=key)

    assert nx.is_directed_acyclic_graph(g) is True

    if lowercase:
        mapping = {s: s.lower() for s in list(g.nodes)}
        g = nx.relabel_nodes(g, mapping)
    return g


def majority_vote(x):
    a, b = np.unique(x, return_counts=True)
    return a[np.argmax(b)]


def majority_count(x):
    _, b = np.unique(x, return_counts=True)
    return np.max(b)

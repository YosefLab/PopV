import networkx as nx

from popv import _utils


def _absolute_accuracy(adata, pred_key, gt_key, save_key=None):
    pred = adata.obs[pred_key].str.lower()
    gt = adata.obs[gt_key].str.lower()

    acc = (pred == gt).astype(int)
    if save_key is not None:
        adata.obs[save_key] = acc
    return acc


def _ontology_accuracy(adata, pred_key, gt_key, obofile, save_key=None):
    G = _utils.make_ontology_dag(obofile, lowercase=True).reverse()
    if not save_key:
        save_key = "ontology_accuracy"
    adata.obs[save_key] = "na"

    def match_type(n1, n2):
        if n1 == n2:
            return "exact"
        elif not set(G.predecessors(n1)).isdisjoint(G.predecessors(n2)):
            return "sibling"
        elif n1 in set(G.predecessors(n2)):
            return "parent"
        elif n2 in set(G.predecessors(n1)):
            return "child"
        else:
            return "no match"

    adata.obs[save_key] = adata.obs.apply(
        lambda x: match_type(x[pred_key], x[gt_key]), axis=1
    )
    return adata.obs[save_key]


def _fine_ontology_sibling_accuracy(adata, obofile, pred_key, gt_key, save_key=None):
    """
    Calculates the fine ontology accuracy and also determines the distance to siblings
    """
    if save_key is None:
        save_key = pred_key + "_ontology_distance"
    adata.obs[save_key] = None

    dag = _utils.make_ontology_dag(obofile, lowercase=True).reverse()

    ontology_distance_dict = {}

    for name, pred_ct, gt_ct in zip(
        adata.obs_names, adata.obs[pred_key], adata.obs[gt_key]
    ):
        pred_ct = pred_ct.lower()
        gt_ct = gt_ct.lower()

        score = None
        combination = f"{pred_ct}_{gt_ct}"
        if combination in ontology_distance_dict:
            score = ontology_distance_dict[combination]
        else:
            if nx.has_path(dag, source=pred_ct, target=gt_ct):
                score = len(nx.shortest_path(dag, source=pred_ct, target=gt_ct)) - 1
            elif nx.has_path(dag, target=pred_ct, source=gt_ct):
                score = len(nx.shortest_path(dag, source=gt_ct, target=pred_ct)) - 1
                score *= -1
            else:
                paths = nx.algorithms.simple_paths.shortest_simple_paths(
                    nx.Graph(dag), source=pred_ct, target=gt_ct
                )
                if len(paths) == 0:
                    score = 1000
                else:
                    score = len(next(paths)) - 1
                    score = str(score) + "_sib"

        ontology_distance_dict[combination] = score
    adata.obs[save_key][name] = score

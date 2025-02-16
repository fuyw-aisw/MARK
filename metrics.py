from typing import Tuple
import numpy as np
from scipy.sparse import base
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cluster
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score 

def calculate_accuracy_and_f1(y_true, y_pred):
    num_classes = max(y_true.max(), y_pred.max()) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    optimal_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([optimal_mapping[label] for label in y_pred])

    accuracy = (y_true == y_pred_aligned).mean()

    f1 = f1_score(y_true, y_pred_aligned, average='macro')

    return accuracy, f1

def true_map_cluster(y_true,y_pred):
    num_classes = max(y_true.max(), y_pred.max()) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    optimal_mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return optimal_mapping
    
def modularity(adjacency, clusters):
    """Computes graph modularity.
    Args:
        adjacency: Input graph in terms of its sparse adjacency matrix.
        clusters: An (n,) int cluster vector.
    Returns:
        The value of graph modularity.
        https://en.wikipedia.org/wiki/Modularity_(networks)
    """
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
    return result / n_edges


def conductance(adjacency, clusters):
    """Computes graph conductance as in Yang & Leskovec (2012).
    Args:
        adjacency: Input graph in terms of its sparse adjacency matrix.
        clusters: An (n,) int cluster vector.
    Returns:
        The average conductance value of the graph clusters.
    """
    inter = 0  # Number of inter-cluster edges.
    intra = 0  # Number of intra-cluster edges.
    cluster_indices = np.zeros(adjacency.shape[0], dtype=bool)
    for cluster_id in np.unique(clusters):
        cluster_indices[:] = 0
        cluster_indices[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_indices, :]
        inter += np.sum(adj_submatrix[:, cluster_indices])
        intra += np.sum(adj_submatrix[:, ~cluster_indices])
    return intra / (inter + intra)
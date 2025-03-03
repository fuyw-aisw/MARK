import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from datetime import datetime
from torch_geometric.utils import to_dense_adj
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from magi.utils import setup_seed, get_sim, get_mask, scale, clustering
from sklearn.metrics.pairwise import pairwise_distances

def read_jsonl(file_path):
    """
    Read a .jsonl file and return the contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the .jsonl file to be read.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

def read_json(file_path):
    """
    Read a .jsonl file and return the contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the .jsonl file to be read.

    Returns:
    list: A list of dictionaries, each representing a JSON object.
    """
    data = []
    with open(file_path) as file:
        data=json.load(file)
    return data

def generate_chat_input_file(input_text, model_name = 'gpt-4o-mini'):
    jobs = []
    for i, text in enumerate(input_text):
        obj = {}
        obj['input'] = text
        jobs.append(obj)
    return jobs 

def get_k_hop_neighbors(data,node_idx,hop=2):
    edge_index = data.edge_index
    visited = set([node_idx]) 
    current_level = set([node_idx])
    for _ in range(hop):
        next_level = set()
        for u in current_level:
            neighbors = edge_index[1, edge_index[0] == u].tolist()
            for v in neighbors:
                if v not in visited:
                    visited.add(v)
                    next_level.add(v)
        current_level = next_level
    
    visited=list(visited)
    visited.remove(node_idx)
    
    for idx,i in enumerate(visited):
        if isinstance(i,torch.Tensor):
            visited[idx]=i.item()
    return visited

def get_top_k_neighbor_simcse(data,sampled_node_idxs,g_feat,k=2,hop=2):
    neighbor_dict = {}
    for i in tqdm(sampled_node_idxs):
        
        neighbors = get_k_hop_neighbors(data,i,hop)
        if len(neighbors) == 0:
            neighbor_dict[i] = []
        elif len(neighbors) <= k:
            neighbor_dict[i] = neighbors
        else:
            sim_score = []
            for j in neighbors:
                sim_score.append((j,1-cosine(g_feat[i],g_feat[j])))
            sorted_score = sorted(sim_score, key=lambda item: item[1], reverse=True)
            neighbor_dict[i] = [sorted_score[m][0] for m in range(k)]
    return neighbor_dict                       
                   
        

def set_seed_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def get_one_hop_neighbors(data,sampled_node_idxs):
    neighbor_dict = {}
    for center_node_idx in sampled_node_idxs:
        #center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = set(neighbors_in_mask(data.edge_index,center_node_idx))
    return neighbor_dict
    
def get_one_hop_neighbors(data,sampled_node_idxs):
    neighbor_dict = {}
    for center_node_idx in sampled_node_idxs:
        #center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = set(neighbors_in_mask(data.edge_index,center_node_idx))
    return neighbor_dict
    
def neighbors_in_mask(edge_index, node_id):
    row, col = edge_index 
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()
    
def get_modul_mask(data,args):
    x, edge_index, y = data.x, data.edge_index, data.y
    #N, E = data.x.shape[0], int(data.edge_index.shape[1]/2)
    N = int(edge_index.max().item()) + 1
    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
    adj = SparseTensor(row=edge_index[0],col=edge_index[1], sparse_sizes=(N, N))
    adj.fill_value_(1.)
    batch = torch.LongTensor(list(range(N)))
    batch, adj_batch = get_sim(batch, adj, wt=args.wt, wl=args.wl)
    mask = get_mask(adj_batch)
    return mask
    
def top_k_samples(feature,n_cluster,cluster_centers,predict_labels,top_k=1):
    distances = pairwise_distances(feature, cluster_centers)
    if top_k == 1:
        top_confidence_samples = []
        for i in range(n_cluster):
            cluster_indices = np.where(predict_labels == i)[0]
            cluster_distances = distances[cluster_indices, i]

            # Get the indices of the top N samples with the smallest distances
            top_n_indices = cluster_indices[np.argmin(cluster_distances)]
            top_confidence_samples.append(top_n_indices)
    else:
        top_confidence_samples = {}
        for i in range(n_cluster):
            cluster_indices = np.where(predict_labels == i)[0]
            cluster_distances = distances[cluster_indices, i]

            # Get the indices of the top N samples with the smallest distances
            top_n_indices = cluster_indices[np.argsort(cluster_distances)[:top_k]]
            top_confidence_samples[i] = top_n_indices
    return top_confidence_samples
def log(*args):
    print(f'[{datetime.now()}]', *args)
    

def delete_non_tensor_attributes(data):
    for attr_name in data.keys:
        if not isinstance(data[attr_name], torch.Tensor):
            delattr(data, attr_name)
    return data

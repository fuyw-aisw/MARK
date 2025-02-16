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
    
def load_mapping():
    arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus, Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule Learning",
        'Neural_Networks': "Neural Networks",
        'Case_Based': "Case Based",
        'Genetic_Algorithms': "Genetic Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement Learning",
        'Probabilistic_Methods': "Probabilistic Methods"
    }
    wikics_mapping = cora_mapping
    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}
    return arxiv_mapping, citeseer_mapping, pubmed_mapping, cora_mapping,wikics_mapping, products_mapping

def delete_non_tensor_attributes(data):
    for attr_name in data.keys:
        if not isinstance(data[attr_name], torch.Tensor):
            delattr(data, attr_name)
    return data

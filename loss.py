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
from sklearn.metrics.pairwise import pairwise_distances


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N . 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def contrastive_loss(Z_i, Z_j, temperature=0.1):
    """
    Parameters:
        Z_i: (n, d) embeddings of the first set of samples
        Z_j: (n, d) embeddings of the second set of samples
        temperature: scalar, temperature scaling factor for similarity
    """
    # Normalize the embeddings
    Z_i = F.normalize(Z_i, p=2, dim=1)
    Z_j = F.normalize(Z_j, p=2, dim=1)
    
    # Compute the similarity matrix between Z_i and Z_j
    sim_matrix = torch.matmul(Z_i, Z_j.T)  # Shape: (n, n)
    sim_matrix /= temperature  # Apply temperature scaling
    
    # Calculate positive logits (diagonal elements)
    positive_logits = torch.diagonal(sim_matrix)
    
    # Calculate the denominator using logsumexp for numerical stability
    denominator = torch.logsumexp(sim_matrix, dim=1)
    
    # Compute the contrastive loss
    loss = -torch.mean(positive_logits - denominator)
    
    return loss

def info_nce_loss(X, D, C, temperature=0.1):
    """
    Parameters:
        X: (N,d) less confident samples
        D: (k,d) embeddings of typical samples in each cluster
        C: (N,k) one hot label of X
    """
    X = F.normalize(X, p=2, dim=1)
    D = F.normalize(D, p=2, dim=1)
    sim_matrix = torch.matmul(X, D.T)
    sim_matrix /= temperature
    N, k = X.shape[0], D.shape[0]
    device = X.device
    mask = torch.zeros(N, k).to(device)
    for i, pos in enumerate(C):
        mask[i, pos] = 1
    positive_logits = torch.sum(sim_matrix * mask, dim=1)
    denominator = torch.logsumexp(sim_matrix, dim=1)
    loss = -torch.mean(positive_logits - denominator)
    
    return loss
        
def clustering_loss(data, labels):
    num_classes = labels.max() + 1
    y_pred = torch.tensor(np.eye(num_classes)[labels])
    print(y_pred.shape)
    adj = to_dense_adj(data.edge_index)[0]
    degrees = torch.sum(adj,dim=0).reshape(-1,1)
    num_of_nodes = adj.shape[0]
    num_of_edges = torch.sum(degrees) / 2
    #C^T AC
    graph_pooled = torch.matmul(torch.matmul(y_pred.t(), adj),y_pred) #k*k
    norm_left = torch.matmul(y_pred.t(),degrees) #k*1
    norm_right = torch.matmul(degrees.t(),y_pred) #1*k
    norm = torch.matmul(norm_left,norm_right)/ 2 / num_of_edges
    spec_loss = - torch.trace(graph_pooled - norm)/ 2 / num_of_edges
    cluster_sizes = torch.sum(y_pred,dim=0)
    #print(torch.norm(cluster_sizes,2), num_of_nodes,y_pred.shape[1])
    clp_loss = torch.norm(cluster_sizes,2)/ num_of_nodes * torch.sqrt(torch.tensor(float(y_pred.shape[1]))) - 1
    
    return spec_loss,clp_loss

class MultipleNegativesRankingLoss_b(torch.nn.Module):

    def __init__(self, scale: float = 0.05):
        super(MultipleNegativesRankingLoss_b, self).__init__()
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cos_sim(self,a: torch.Tensor, b: torch.Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, embeddings_a, embeddings_b, labels):

        scores = self.cos_sim(embeddings_a, embeddings_b) / self.scale

        #labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)

class MultipleNegativesRankingLoss_a(torch.nn.Module):

    def __init__(self, scale: float = 0.05):
        super(MultipleNegativesRankingLoss_a, self).__init__()
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cos_sim(self,a: torch.Tensor, b: torch.Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, embeddings_a, embeddings_b):

        scores = self.cos_sim(embeddings_a, embeddings_b) / self.scale

        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)

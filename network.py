from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GCNConv, GINConv, SAGEConv
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch_geometric.nn import LabelPropagation
from torch_geometric.nn.models import GAT
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PYGGATConv
import copy
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
import numpy as np
import math
import tqdm
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import torch.distributed as dist
from loss import NT_Xent
from torch_geometric.utils import to_dense_adj,dropout_adj
from sklearn.cluster import KMeans
import numpy as np
from magi.utils import setup_seed, get_sim, get_mask, scale, clustering
from magi.model import Model, Encoder
from loss import MultipleNegativesRankingLoss_a,MultipleNegativesRankingLoss_b 
from torch_geometric.data import Data
from metrics import true_map_cluster
BIG_CONSTANT = 1e8

def get_model(args):
    if args.model_name == 'MLP':
        return UniversalMLP(args.num_layers, args.input_dim, args.hidden_dimension, args.hidden_dimension, args.dropout, args.norm, args.return_embeds)
    elif args.model_name == 'GCN':
        return GCN(args.num_layers, args.input_dim, args.hidden_dimension, args.hidden_dimension, args.dropout, args.norm)
    elif args.model_name == 'MLP2':
        return DeepMLP(args.input_dim, args.hidden_dimension)
    elif args.model_name == 'SAGE':
        return SAGE(args.input_dim, args.hidden_dimension, args.hidden_dimension, args.num_layers, args.dropout)
    elif args.model_name == 'GAT':
        return GAT2(args.input_dim, args.hidden_dimension, args.num_layers, args.hidden_dimension, args.dropout, args.dropout, args.num_of_heads, args.num_of_out_heads, args.norm)


class GAT2(torch.nn.Module):
    def __init__(self, num_feat, hidden_dimension, num_layers, num_class, dropout, attn_drop, num_of_heads = 1, num_of_out_heads = 1, norm = None):
        super().__init__()
        self.layers = []
        self.bns = []
        if num_layers == 1:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = False, dropout=attn_drop)
        else:
            self.conv1 = PYGGATConv(num_feat, hidden_dimension, num_of_heads, concat = True, dropout=attn_drop)
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))
        self.layers.append(self.conv1)
        for _ in range(num_layers - 2):
            self.layers.append(
                PYGGATConv(hidden_dimension * num_of_heads, hidden_dimension, num_of_heads, concat = True, dropout = dropout)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dimension * num_of_heads))

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        if num_layers > 1:
            self.layers.append(PYGGATConv(hidden_dimension * num_of_heads, num_class, heads=num_of_out_heads,
                             concat=False, dropout=attn_drop).cuda())
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList(self.bns)
        self.norm = norm 
        self.num_layers = num_layers
        self.with_bn = True if self.norm == 'BatchNorm' else False
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.layers[i](x, edge_index)
            if i != self.num_layers - 1:
                if self.with_bn:
                    x = self.bns[i](x)
                x = F.elu(x)
        return x



class GATWrapper(torch.nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, out_size, dropout):
        super().__init__()
        self.gat = GAT(in_size, hidden_size, num_layers, out_size, dropout)
    
    def forward(self, data):
        x, edge_index= data.x, data.edge_index
        return self.gat(x, edge_index)



class UniversalMLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None, return_embeds = False) -> None:
        super().__init__()
        hidden_dimensions = [hidden_dimension] * (num_layers - 1)
        self.hidden_dimensions = [input_dim] + hidden_dimensions + [num_classes]
        self.mlp = MLP(channel_list=self.hidden_dimensions, dropout=dropout, norm=norm)
        self.return_embeds = False

    def forward(self, data):
        x = data.x
        return self.mlp(x)
    
    def inference(self, x_all, subgraph_loader, device):
        xs = []
        for batch in tqdm.tqdm(subgraph_loader):
            edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
            edge_index = edge_index.to(device)
            # import ipdb; ipdb.set_trace()
            x = x_all[n_id][:batch.batch_size].to(device)
            x = self.mlp(x)
            xs.append(x.cpu())
        x_all = torch.cat(xs, dim=0)
        return x_all


class DeepMLP(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_size, 1024),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.LayerNorm(512),
                nn.Linear(512, out_size),
        )
    
    def forward(self, data):
        x = data.x
        return self.mlp(x)




class GCN(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dimension, num_classes, dropout, norm=None) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, num_classes, cached=False,
                             normalize=True))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dimension, cached=False,
                             normalize=True))
            if norm:
                self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
            else:
                self.norms.append(torch.nn.Identity())

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dimension, hidden_dimension, cached=False,
                             normalize=True))
                if norm:
                    self.norms.append(torch.nn.BatchNorm1d(hidden_dimension))
                else:
                    self.norms.append(torch.nn.Identity())

            self.convs.append(GCNConv(hidden_dimension, num_classes, cached=False, normalize=True))

    def forward(self, data):
        x, edge_index, edge_weight= data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                edge_index, n_id, size = batch.edge_index, batch.n_id, batch.batch_size
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all

    

## backbone:dmon
class Graph_embed_dmon(torch.nn.Module):
    def __init__(self, args):
        super(Graph_embed_dmon, self).__init__()
        self.model = get_model(args)
        self.cluster_size = args.num_classes
        self.args = args
        #self.lin = nn.Linear(args.hidden_dimension,args.num_classes)

    def forward(self,data):
        #batch_size = data.batch_size
        x,edge_index = data.x, data.edge_index
        # aug1
        random_noise1 = torch.rand_like(x)
        x_aug1 = x + torch.sign(x) * F.normalize(random_noise1, dim=-1) * 0.1 
        edge_index_aug1 = dropout_adj(edge_index, p=0.2)[0]
        data_aug1 = Data(x=x_aug1, edge_index=edge_index_aug1)
        # aug2
        random_noise2 = torch.rand_like(x)
        x_aug2 = x + torch.sign(x) * F.normalize(random_noise2, dim=-1) * 0.1 
        edge_index_aug2 = dropout_adj(edge_index, p=0.2)[0]
        data_aug2 = Data(x=x_aug2, edge_index=edge_index_aug2) 
        
        #g_feat1 = self.model(data)
        g_feat1 = self.model(data_aug1)#[:batch_size]
        g_feat2 = self.model(data_aug2)#[:batch_size]
        
        
        return g_feat1,g_feat2,data_aug1,data_aug2
    
    @staticmethod
    def find_mismatch_nodes(y_pred1,y_pred2):
        mapping = true_map_cluster(y_pred2,y_pred1) #y_true,y_pred mapping cluster
        #print(mapping)
        y_pred_label2 = [mapping[int(label)] for label in y_pred2]
        y_pred_label1 = y_pred1
        mis_mask = y_pred_label1 != y_pred_label2
        mis_mask = np.nonzero(mis_mask)[0]
        
        return mis_mask 
    
    def cluster(self,g_feat1,g_feat2,seed=1):
        y_pred1 = clustering(g_feat1.detach().cpu().numpy(), self.cluster_size, "cuda", self.args.device, seed)#, spectral_clustering=True)
        y_pred2 = clustering(g_feat2.detach().cpu().numpy(), self.cluster_size, "cuda", self.args.device, seed)#, spectral_clustering=True)
        mis_mask = self.find_mismatch_nodes(y_pred1,y_pred2)

        return y_pred1,y_pred2,mis_mask
    
    def dmon_loss(self,data,labels,device,alpha=0.1):
        num_classes = labels.max() + 1
        y_pred = torch.tensor(np.eye(num_classes)[labels], dtype=torch.float32).float().to(device).requires_grad_()
        adj = to_dense_adj(data.edge_index, max_num_nodes=y_pred.shape[0])[0]
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
        loss = spec_loss + alpha * clp_loss
        return loss
                  
    def cl_loss1(self,g_feat1,g_feat2,tau):
        criterion = MultipleNegativesRankingLoss_a(scale=tau)
        loss = criterion(g_feat1,g_feat2)
        return loss
       
    
    def cl_loss2(self,g_feat,p_feat,labels,tau):
        criterion = MultipleNegativesRankingLoss_b(scale=tau)
        loss = criterion(g_feat,p_feat,labels)
        return loss

# backbone: magi    

class Graph_embed_magi(torch.nn.Module):
    def __init__(self, args, num_features=384):#768):#1433)
        super(Graph_embed_magi, self).__init__()
        hidden = list(map(int, args.hidden.split(',')))
        self.encoder = Encoder(num_features, hidden, base_model=GCNConv,dropout=args.dropout, ns=args.ns)
        self.model = Model(self.encoder, in_channels=hidden[-1], project_hidden=None, tau=args.tau)
        self.cluster_size = args.num_classes
        self.args = args

    def forward(self,data):
        x,edge_index = data.x, data.edge_index
        # aug1
        random_noise1 = torch.rand_like(x)
        x_aug1 = x + torch.sign(x) * F.normalize(random_noise1, dim=-1) * 0.1 
        edge_index_aug1 = dropout_adj(edge_index, p=0.2)[0]
        data_aug1 = Data(x=x_aug1, edge_index=edge_index_aug1)
        # aug2
        random_noise2 = torch.rand_like(x)
        x_aug2 = x + torch.sign(x) * F.normalize(random_noise2, dim=-1) * 0.1 
        edge_index_aug2 = dropout_adj(edge_index, p=0.2)[0]
        data_aug2 = Data(x=x_aug2, edge_index=edge_index_aug2) 
        
        g_feat1 = self.model(x_aug1, edge_index_aug1) 
        g_feat2 = self.model(x_aug2, edge_index_aug2)
        g_feat1 = F.normalize(scale(g_feat1), p=2, dim=1)
        g_feat2 = F.normalize(scale(g_feat2), p=2, dim=1)
        #mis_mask = (1-(y_pred1==y_pred2)).float().bool().cpu().detach()
        #mis_mask = self.find_mismatch_nodes(y_pred1,y_pred2)
        #data_aug = Data(x=x_aug,edge_index=edge_index_aug)
        return g_feat1,g_feat2,data_aug1,data_aug2
    
    def cluster(self,g_feat1,g_feat2,seed=1):
        y_pred1 = clustering(g_feat1.detach().cpu().numpy(), self.cluster_size, "cuda", self.args.device, seed)#, spectral_clustering=True)
        y_pred2 = clustering(g_feat2.detach().cpu().numpy(), self.cluster_size, "cuda", self.args.device, seed)#, spectral_clustering=True)
        mis_mask = self.find_mismatch_nodes(y_pred1,y_pred2)

        return y_pred1,mis_mask

    def magi_loss(self,g_feat,mask):
        loss = self.model.loss(g_feat, mask)
        return loss
    
    def cl_loss1(self,g_feat1,g_feat2,tau):
        criterion = MultipleNegativesRankingLoss_a(scale=tau)
        loss = criterion(g_feat1,g_feat2)
        return loss
       
    
    def cl_loss2(self,g_feat,p_feat,labels,tau):
        criterion = MultipleNegativesRankingLoss_b(scale=tau)
        loss = criterion(g_feat,p_feat,labels)
        return loss
                 
    @staticmethod
    def find_mismatch_nodes(y_pred1,y_pred2):
        #y_pred_label1 = torch.argmax(y_pred1,dim=1)
        #y_pred_label2 = torch.argmax(y_pred2,dim=1)
        #print(y_pred_label1)
        mapping = true_map_cluster(y_pred2,y_pred1) #y_true,y_pred mapping cluster
        y_pred_label2 = [mapping[int(label)] for label in y_pred2]
        y_pred_label1 = y_pred1
        mis_mask = y_pred_label1 != y_pred_label2
        mis_mask = np.nonzero(mis_mask)[0]

        return mis_mask 
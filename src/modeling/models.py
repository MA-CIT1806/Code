import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, TAGConv, global_add_pool, graclus, max_pool, avg_pool, JumpingKnowledge, GlobalAttention, GATConv
from torch_geometric.data import Batch
from torch_geometric.utils import dropout_adj
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ELU, GRU

import numpy as np
from src.modeling.layers import *


class GIN(nn.Module):
    """
    Model for comparison on graph classification tasks.
    Corresponding paper: https://arxiv.org/abs/1810.00826
    """

    def __init__(self, args):
        super(GIN, self).__init__()
        
        num_node_features = args.get("num_node_features")
        num_hidden = args.get("num_hidden")
        num_classes = args.get("num_classes")
        self.dropout = args.get("dropout", 0.0)

        nn1 = Sequential(Linear(num_node_features, num_hidden), LeakyReLU(), Linear(num_hidden, num_hidden))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(num_hidden)

        nn2 = Sequential(Linear(num_hidden, num_hidden), LeakyReLU(), Linear(num_hidden, num_hidden))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(num_hidden)
        
        self.fc1 = Linear(num_hidden, num_hidden)
        self.fc2 = Linear(num_hidden, num_classes)
        
        self.graph_embedding_function = args.get("graph_embedding_function", None) 
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.bn2(x)
    
        x = global_add_pool(x, batch)
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 0)
            
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 1)
        
        return F.log_softmax(x, dim=-1)
##############################################
class GCN(torch.nn.Module):
    """
    Model for comparison on graph classification tasks.
    Corresponding paper: https://arxiv.org/abs/1609.02907
    """

    def __init__(self, args):
        super(GCN, self).__init__()
        
        num_node_features = args.get("num_node_features")
        num_classes = args.get("num_classes")
        self.dropout = args.get("dropout", 0.0)
        
        self.norm = NormalizeFeatures()
        
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        
        self.fc1 = Linear(32, 32)
        self.fc2 = Linear(32, num_classes)
        
        self.graph_embedding_function = args.get("graph_embedding_function", None) 
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        data = self.norm(data)
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        x = global_add_pool(x, batch)
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 0) 
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)      
        x = self.fc2(x)
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 1) 

        return F.log_softmax(x, dim=1)
##############################################
##############################################
class Proposal(nn.Module):
    """
    Proposed model architecture of thesis.
    Exploits both the spatial and temporal dimension.
    """

    def __init__(self, args):
        super(Proposal, self).__init__()

        self.args = args

        num_node_features = args.get("num_node_features")
        num_hidden = args.get("num_hidden")
        num_flex = args.get("num_flex")
        num_classes = args.get("num_classes")
        dropout = args.get("dropout", 0.0)
        
        self.dropout = dropout
        
        self.pe = PositionalEncoding(23, dropout)
        
        self.classic_conv = nn.Conv1d(23, 23, 3, padding=1)
        self.classic_conv_bn = torch.nn.BatchNorm1d(num_node_features)
        
        self.ll1 = Linear(num_node_features, num_hidden)
        
        self.att_block1 = AttentionBlock(num_hidden, 8*num_hidden, depth=5)
                
        self.gp = GlobalAttention(
            torch.nn.Sequential(Linear(8*num_hidden, 21*num_hidden), Linear(21*num_hidden, 1))
        )
        
        self.slc4 = SublayerConnection(8*num_hidden)
     
        self.fff = FinalFeedForward(8*num_hidden, num_flex, dropout)
        
        self.classic_conv2 = nn.Conv1d(1, 10, 9, padding=4)
        self.classic_conv_bn2 = torch.nn.BatchNorm1d(8*num_hidden)  
        
        self.fc_task1 = Linear(8*num_hidden, num_node_features) # random masked node task
        self.fc_task2 = Linear(8*num_hidden, num_node_features) # random node in next graph task
        
        self.fc1 = Linear(8*num_hidden, num_classes)
        
        self.graph_embedding_function = args.get("graph_embedding_function", None)    
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)    

    def prepare_fine_tuning(self):
        num_hidden = self.args.get("num_hidden")
        num_classes = self.args.get("num_classes")

        self.fc1 = Linear(8*num_hidden, num_classes)
        
        return []                 
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        bs = torch.unique(batch).size(0)
        
        # positional encoding
        x = x.view(bs, 23 , -1).permute(0, 2, 1)
        x = self.pe(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs*23, -1)
        
        # apply classic convolution
        x = x.reshape(bs, 23 , -1)
        x = self.classic_conv(x)
        x = x.reshape(bs*23, -1)
        x = self.classic_conv_bn(x)
               
        # dropout adjacency matrix
        edge_index, _ = dropout_adj(edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        
        x = self.ll1(x)
        
        # att block 1
        x = self.att_block1(x, edge_index)
                
        x = self.slc4(x, self.fff)
        
        x = self.gp(x, batch)
        
        # apply classic convolution
        x = x.unsqueeze(1)
        x = self.classic_conv2(x)
        x = x.mean(axis=1)
        x = self.classic_conv_bn2(x)        
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 0)          
        
        x = self.fc1(x)     
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 1)         
         
        return F.log_softmax(x, dim=1)
        
###############################################
###############################################    
class AttentionBlock(nn.Module):
    """
    Scalable attention blocks. Combines TAGCN, GAT and JK.
    TAGCN exploits the spatial dimension with the help of the adjacency matrix.
    GAT exploits the spatial dimension by performing attention mechanisms on node features.
    Finally, JK learns a weighted combination of graph layer outputs.
    """

    def __init__(self, 
                 in_channels,
                 out_channels,
                 depth=3,
                 jk_depth=7,
                 base_conv=TAGConv, 
                 att_conv=GATConv, 
                 base_conv_settings={"K": 3},
                 att_conv_settings={}):
        super(AttentionBlock, self).__init__()
        
        assert out_channels % in_channels == 0, "out_channels must be a multiple of in_channels"
        
        ratio = out_channels // in_channels
        
        att_conv_settings = dict(**att_conv_settings, heads=ratio)
        
        self.base_conv_list = torch.nn.ModuleList([])
        self.att_conv_list = torch.nn.ModuleList([])
        self.slc_list = torch.nn.ModuleList([])
        
        self.jk = JumpingKnowledge("lstm", ratio*in_channels, jk_depth)
        
        for _ in range(depth):
            self.slc_list.append(SublayerConnection(in_channels))
            
            self.base_conv_list.append(base_conv(in_channels, in_channels, **base_conv_settings))
            self.att_conv_list.append(att_conv(in_channels, in_channels, **att_conv_settings))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)                 
            
    def forward(self, x, edge_index):
        
        jk_list = []
        
        for i in range(len(self.base_conv_list)):
            x = self.slc_list[i](x, self.base_conv_list[i], edge_index=edge_index)
            jk_list.append(self.att_conv_list[i](x, edge_index))
        
        x = self.jk(jk_list)
        
        return x
        
class FinalFeedForward(nn.Module):
    """
    Simple feed-forward network. Utilizes two linear layer and dropout in between.
    """
    
    def __init__(self, num_hidden, num_flex, dropout=0.5):
        super(FinalFeedForward, self).__init__()
        
        self.fc1 = Linear(num_hidden, num_flex)
        self.fc2 = Linear(num_flex, num_hidden)
        
        self.dropout = dropout
        
    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)      
        x = self.fc2(x)
        
        return x
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        indices = np.arange(1, pe.shape[1], 2)
        pe[:, indices] = torch.cos(position * div_term)[:, :len(indices)]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, act=None):
        super(SublayerConnection, self).__init__()
        self.norm = torch.nn.BatchNorm1d(size, affine=False)
        self.act = act
        if self.act is not None:
            print("using activation-function...")
        
    def forward(self, x, sublayer, edge_index=None):
        "Apply residual connection to any sublayer with the same size."
        layer_out = None
        
        if edge_index is not None:
            layer_out = sublayer(x, edge_index)
        else:
            layer_out = sublayer(x)
            
        layer_out = self.act(layer_out) if self.act is not None else layer_out
        
        return self.norm(x + layer_out)
###########################################################
###########################################################
##### helper functions for local pooling operations #######    

def apply_sag_pooling(instance, x, edge_index, batch):
    x, edge_index, _, batch,_, _ = instance(x, edge_index, batch=batch)
    return x, edge_index, batch


def apply_edge_pooling(instance, x, edge_index, batch):
    x, edge_index, batch,_ = instance(x, edge_index, batch)
    return x, edge_index, batch

def apply_graclus_pooling(x, edge_index, batch, method="max"):
        cluster = graclus(edge_index)
        func = max_pool if method == "max" else avg_pool
        new_data = func(cluster, Batch(x=x, edge_index=edge_index, batch=batch))
        return new_data.x, new_data.edge_index, new_data.batch       
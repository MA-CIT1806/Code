import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Batch
from torch_geometric.utils import dropout_adj
from torch.nn import Sequential, Linear, ELU, GRU

import numpy as np
from src.modeling.layers import *
from src.modeling.models_helpers import prepare_task_learning
from src.modeling.models import * # !!!

CUDA_0 = "cuda:0"
CUDA_1 = "cuda:1"

##############################################
##############################################
class ProposalRL(nn.Module):
    def __init__(self, args):
        super(ProposalRL, self).__init__()

        num_node_features = args.get("num_node_features")
        num_hidden = args.get("num_hidden")
        num_flex = args.get("num_flex")
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
        
        self.graph_embedding_function = args.get("graph_embedding_function", None)    
        
        self.node_counter = {}
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def count_nodes(self, masked_nodes_prim):
        for node in masked_nodes_prim:
            if node in self.node_counter:
                self.node_counter[node] += 1
            else:
                self.node_counter[node] = 1
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        next_possible_nodes, headers = data.next_possible_nodes, data.headers
        
        bs = torch.unique(batch).size(0)

        ### SETUP TASK LEARNING ###
        x, edge_index, task1_true, task2_true, mask, masked_nodes_prim = prepare_task_learning(x, next_possible_nodes, headers, edge_index, bs)
        if self.training:
            self.count_nodes(masked_nodes_prim)
        ########################### 
        
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
        
        x = self.gp(x[mask], batch[mask])
        
        # apply classic convolution
        x = x.unsqueeze(1)
        x = self.classic_conv2(x)
        x = x.mean(axis=1)
        x = self.classic_conv_bn2(x)                     

        task1_pred = self.fc_task1(x) # random masked node task
        task2_pred = self.fc_task2(x) # random node in next graph task         
            
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 0)            
            
        return x, task1_true, task1_pred, task2_true, task2_pred  
##############################################
##############################################
class ProposalRLLL(nn.Module):
    def __init__(self, args):
        super(ProposalRLLL, self).__init__()
        
        self.args = args
        
        self.sub_module1 = SubModule1(args)
        
        self.fc1 = None
        
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
        
        return [self.fc1]
        
    def forward(self, data):
        
        x = self.sub_module1(data)
                        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 0)
       
        x = self.fc1(x)     
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 1)         
         
        return F.log_softmax(x, dim=1)
############################################################################
class ProposalRLGRU(nn.Module):
    def __init__(self, args):
        super(ProposalRLGRU, self).__init__()
        
        self.args = args
        self.dropout = args.get("dropout", 0.5)
        
        self.sub_module1 = SubModule1(args).to(CUDA_1)
        
        self.bn1 = None
        self.sub_module2 = None
        self.fc1 = None
        
        self.graph_embedding_function = args.get("graph_embedding_function", None)    
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)   
                
    def prepare_fine_tuning(self):
        num_hidden = self.args.get("num_hidden")
        num_classes = self.args.get("num_classes")
        
        self.sub_module1 = self.sub_module1.to(CUDA_1)
        
        self.bn1 = torch.nn.BatchNorm1d(8*num_hidden, affine=False).to(CUDA_0)
        self.sub_module2 = SubModule2(self.args).to(CUDA_0)
        self.fc1 = Linear(8*num_hidden, num_classes).to(CUDA_0)
        
        return [self.bn1, self.sub_module2, self.fc1]                
        
    def forward(self, data):
        x = self.sub_module1(data)
                        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 0)
            
        x = x.to("cuda:0")
        
        x = self.bn1(x)
        x = self.sub_module2(x) 

        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 1)          
                
        x = self.fc1(x)
        
        if self.graph_embedding_function is not None:
            self.graph_embedding_function(x, 2)         
        
        x = x.to("cuda:1")
        return F.log_softmax(x, dim=1)
    
################## SUB-MODULES FOR TF #################################
#######################################################################
class SubModule1(nn.Module):
    def __init__(self, args):
        super(SubModule1, self).__init__()
        
        num_node_features = args.get("num_node_features")
        num_hidden = args.get("num_hidden")
        num_flex = args.get("num_flex")
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
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)         
        
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
        
        return x
    
class SubModule2(nn.Module):
    def __init__(self, args):
        super(SubModule2, self).__init__()
        
        num_hidden = args.get("num_hidden")
        self.num_hidden = num_hidden
        dropout = args.get("dropout", 0.5)
        
        self.device = CUDA_0
        self.num_layers = args.get("num_layers", 2)
        self.bidirectional = args.get("bidirectional", True)
        
        self.dropout = dropout
        
        self.gru = torch.nn.GRU(input_size=8*num_hidden, 
                        hidden_size=4*num_hidden,
                        num_layers=self.num_layers, 
                        dropout=dropout, 
                        batch_first=True,
                        bidirectional=self.bidirectional)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        
        ### GRU Start ###
        num_directions = 2 if self.bidirectional else 1
        x = torch.unsqueeze(x, dim=0) # "create" batch
        h_0 = self._init_hidden(num_directions, x)
        x, _ = self.gru(x, h_0)
        x = torch.squeeze(x, dim=0) # "delete" batch
        ### GRU End ###
        
        return x
        
    def _init_hidden(self, num_directions, x):
        return torch.zeros((num_directions * self.num_layers, x.size(0), 4*self.num_hidden), dtype=torch.float32,
                         device=self.device)     
    
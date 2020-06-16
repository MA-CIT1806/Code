import math
import scipy.sparse as sp
import numpy as np

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch_geometric.utils import get_laplacian, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing, GlobalAttention
from torch_geometric.nn.inits import glorot, zeros, uniform

from src.modeling.utils import *       


import torch
from torch_scatter import scatter_add
from torch_geometric.utils import softmax

    
class AGCNConv(MessagePassing):
    """
    own implementaion of https://arxiv.org/abs/1801.03226
    """
    
    def __init__(self, in_channels, out_channels, k, bias=True, sigma=3, **kwargs):
        
        super(AGCNConv, self).__init__(aggr='add', **kwargs)
                
        self.pi = torch.tensor([torch.acos(torch.zeros(1)).item() * 2]).to("cuda:1") # not defined in pytorch
        self.k = k
        self.sigma = torch.tensor([sigma]).to("cuda:1")
        self.eye = torch.eye(in_channels).to("cuda:1")
        
        ## below: trainable ##
        self.matrix_w = Parameter(self.create_non_singular_matrix(in_channels))
        self.alpha = Parameter(torch.Tensor(1,))
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)        
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias) 
        uniform(1, self.alpha) 
        
    @staticmethod
    def create_non_singular_matrix(in_channels):
        result = torch.rand(in_channels, in_channels)
        index = 0
        while torch.det(torch.matmul(result, result.t())) < 10:
            result = torch.rand(in_channels, in_channels)
            index += 1
            
        return result    
    
    @staticmethod
    def calculate_laplacian(edge_index, num_nodes, edge_weight, dtype=None):
        
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                'sym', dtype,
                                                num_nodes)
        
        edge_weight[edge_weight == float('inf')] = 0
        
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1,
                                                 num_nodes=num_nodes)

        return edge_index, edge_weight    
    
    def _adapt_adjacency_matrix(self, x, edge_index):
        
        matrix_m = torch.matmul(self.matrix_w, self.matrix_w.T)
                        
        differences = (x.unsqueeze(1) - x.unsqueeze(0))[edge_index[0], edge_index[1], :] 
        
        distances = torch.matmul(torch.matmul(differences, matrix_m), differences.t())
        distances = torch.diag(distances)  
        distances = torch.sqrt(distances)
        
        gaussian_kernel = torch.exp(-(distances / (2*(self.sigma**2))))
        
        return (1 / (self.sigma * torch.sqrt(2*self.pi))) * gaussian_kernel # normalize gaussian kernel
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
                
        # get original graph laplacian
        laplacian = self.calculate_laplacian(edge_index, x.size(self.node_dim), edge_weight, dtype=x.dtype)
                    
        # calculate adapted adjacency matrix
        edge_weight = self._adapt_adjacency_matrix(x, edge_index)
        
        # learn residual graph laplacian
        res_laplacian = self.calculate_laplacian(edge_index, x.size(self.node_dim), edge_weight, dtype=x.dtype)       
        
        # learn optimal graph laplacian, extract weights from optimal graph laplacian?
        opt_laplacian_edge_weight = laplacian[1] + (self.alpha * res_laplacian[1])         
        
        tx_0 = x
        out = tx_0

        if self.k > 1:
            tx_1 = self.propagate(edge_index=res_laplacian[0], x=x, norm=opt_laplacian_edge_weight)
            out = tx_1

        for _ in range(2, self.k):
            tx_2 = 2 * self.propagate(edge_index=res_laplacian[0], x=tx_1, norm=opt_laplacian_edge_weight) - tx_0
            out = tx_2
            tx_0, tx_1 = tx_1, tx_2
            
        out = torch.matmul(out, self.weight)    

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.k)
    
       
class SACNConv(MessagePassing):
    """
    own implementation of https://papers.nips.cc/paper/7287-structure-aware-convolutional-neural-networks
    """

    def __init__(self, in_channels, k, bias=True, **kwargs):
        super(SACNConv, self).__init__(aggr='add', **kwargs)

        assert k > 0

        self.in_channels = in_channels

        self.relationships_M = Parameter(
            uniform((in_channels, in_channels)))

        self.chebyshev_coefficients = Parameter(
            uniform((k,)))

    def _create_relationship_R(self, x, edge_index):
                
        relationship_matrix = torch.mm(
            torch.mm(x, self.relationships_M), x.t())
        
        return torch.tanh(relationship_matrix[edge_index[0], edge_index[1]])

    def _create_functional_F(self, relationship_r, edge_index):
        """
        Approximation of functional Filter by truncated expansion of Chebyshev polynomials
        """
        t_k_list = chebyshev_polynomials(
            relationship_r, self.chebyshev_coefficients.size(0))
        
        edge_weights = (
            self.chebyshev_coefficients[:, None]*torch.stack(t_k_list)).sum(0)

        return edge_weights

    def forward(self, x, edge_index, edge_weight=None, batch=None):
                
        relationship_r = self._create_relationship_R(x, edge_index)

        weights = self._create_functional_F(relationship_r, edge_index)
        
        return self.propagate(edge_index, x=x, weights=weights)

    def message(self, x_j, weights):
        return weights.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, K={})'.format(
            self.__class__.__name__, self.in_channels,
            self.chebyshev_coefficients.size(0))


class LAGNConv(MessagePassing):
    """
    own implementation of https://ieeexplore.ieee.org/abstract/document/8709773
    """

    def __init__(self, in_channels, out_channels, k, linear_transform=False, bias=True, **kwargs):
        super(LAGNConv, self).__init__(aggr='add', **kwargs)

        num_basis_functions = kwargs.get("num_basis_functions", 10)

        assert k > 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.inner_coefficients = Parameter(uniform((k, num_basis_functions)))
        self.outer_coefficients = Parameter(uniform((k, num_basis_functions)))
        
        self.lin = lambda x: x
        if linear_transform:
            self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)
        
    def _coefficients_mul_matrices(self, coefficients, x):
        
        return(coefficients[:, None, None]*x).sum(0)
        
    def _compute_chebyshev_polynomials(self, X, k):
        """
        Approximation desired function by truncated expansion of Chebyshev polynomials
        """
        t_k_list = chebyshev_polynomials(torch.tanh(X), k, kind=2)
        
        return torch.stack(t_k_list)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        
        inner_approx = self._compute_chebyshev_polynomials(x, self.inner_coefficients.size(1))
                
        final_result = []
        for k in range(self.inner_coefficients.size(0)):
            inner_result = self.propagate(edge_index, x=self._coefficients_mul_matrices(self.inner_coefficients[k], inner_approx))
            outer_approx = self._compute_chebyshev_polynomials(inner_result, self.outer_coefficients.size(1))
            outer_result = self._coefficients_mul_matrices(self.outer_coefficients[k], outer_approx)
            final_result.append(outer_result)
        
        t_sum = torch.stack(final_result).sum(0)
        return self.lin(t_sum)
  
    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.inner_coefficients.size(0))

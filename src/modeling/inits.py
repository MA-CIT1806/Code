import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.init import uniform_
from src.modeling.utils import sparse_mx_to_torch_sparse_tensor


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    tensor = torch.empty(shape)
    uniform_(tensor, a=-init_range, b=init_range)
    return tensor


def uniform(shape, low=-0.1, high=0.1):
    tensor = torch.empty(shape)
    uniform_(tensor, a=low, b=high)
    return tensor

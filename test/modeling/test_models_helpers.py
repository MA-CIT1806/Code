import pytest
import torch

from src.modeling.models_helpers import node_probas, extend_task1_indices

def test_extend_task1_indices():

    markers = torch.arange(0,230, 23)
    indices = [3]*23
    headers = [list(node_probas.keys())] * 23

    ext_indices = extend_task1_indices(markers, indices, headers)
    
    for m in markers:
        assert all([m+3+i in ext_indices for i in range(3)])
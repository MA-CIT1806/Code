import torch
from torch_geometric.utils import subgraph
import numpy as np

related_nodes = {
    'block/allocation': ['block/allocation'],
    'block/capacity': ['block/capacity'],
    'block/physical': ['block/physical'],
    'cpu': ['cpu', 'cpu/system', 'cpu/user', 'cpu/virt','general/cpu'],
    'cpu/system': ['cpu', 'cpu/system', 'cpu/user', 'cpu/virt','general/cpu'],
    'cpu/user': ['cpu', 'cpu/system', 'cpu/user', 'cpu/virt','general/cpu'],
    'cpu/virt': ['cpu', 'cpu/system', 'cpu/user', 'cpu/virt','general/cpu'],
    'disk-io/all/io': ['disk-io/all/io'],
    'disk-io/all/ioBytes': ['disk-io/all/ioBytes'],
    'general/cpu': ['cpu', 'cpu/system', 'cpu/user', 'cpu/virt','general/cpu'],
    'general/maxMem': ['general/maxMem'],
    'general/mem': ['general/mem'],
    'mem/available': ['mem/available', 'mem/percent', 'mem/used'],
    'mem/percent': ['mem/available', 'mem/percent', 'mem/used'], 
    'mem/used': ['mem/available', 'mem/percent', 'mem/used'],
    'net-io/bytes': ['net-io/bytes'],
    'net-io/dropped': ['net-io/dropped'], 
    'net-io/errors': ['net-io/errors'], 
    'net-io/packets': ['net-io/packets'], 
    'net-io/rx_bytes': ['net-io/rx_bytes'],
    'net-io/rx_packets': ['net-io/rx_packets'], 
    'net-io/tx_bytes': ['net-io/tx_bytes'], 
    'net-io/tx_packets': ['net-io/tx_packets']
}

node_probas = {
    'block/allocation': 1 / 23,
    'block/capacity': 1 / 23,
    'block/physical': 1 / 23,
    'cpu':  5 / 23,
    'cpu/system':  5 / 23,
    'cpu/user':  5 / 23,
    'cpu/virt':  5 / 23,
    'disk-io/all/io': 1 / 23,
    'disk-io/all/ioBytes': 1 / 23,
    'general/cpu':  5 / 23,
    'general/maxMem': 1 / 23,
    'general/mem': 1 / 23,
    'mem/available': 3 / 23,
    'mem/percent':  3 / 23, 
    'mem/used':  3 / 23,
    'net-io/bytes': 1 / 23,
    'net-io/dropped': 1 / 23, 
    'net-io/errors': 1 / 23, 
    'net-io/packets': 1 / 23, 
    'net-io/rx_bytes': 1 / 23,
    'net-io/rx_packets': 1 / 23, 
    'net-io/tx_bytes': 1 / 23, 
    'net-io/tx_packets': 1 / 23
}

def extend_task1_indices(markers, indices, headers):
    ext_indices = []
    for idx, i in enumerate(markers):
        corresponding_headers = headers[idx]
        related_headers = related_nodes[corresponding_headers[indices[idx]]]
        ext_indices += [i + j for j, el in enumerate(corresponding_headers) if el in related_headers] 
    
    ext_indices = torch.LongTensor(ext_indices, device=markers.device)
    return ext_indices


def sample_indices(headers, device):
    probas = list(node_probas.values())
    probas = [el / sum(probas) for el in probas]
    indices = np.random.choice(23, len(headers), p=probas)   
    
    node_list = list(node_probas.keys())
    node_names = [node_list[i] for i in indices]
    
    indices_list = []
    for idx, h in enumerate(headers):
        for i, n in enumerate(h):
            if n == node_names[idx]:
                indices_list.append(i)
                break
                
    return torch.LongTensor(indices_list, device=device).reshape(-1, 1), node_names            
        
def prepare_task_learning(x, next_possible_nodes, headers, edge_index, batch_size):

    markers = torch.arange(0, (batch_size*23), 23).reshape(batch_size, 1)
    
    # prepare task 1: predict random masked node in graph
    #random_indices_task1 = torch.randint(0, 23, (batch_size, 1))
    random_indices_task1, masked_nodes_prim = sample_indices(headers, markers.device)
    indices_task1 = markers + random_indices_task1
    task1_true = x[indices_task1].squeeze(1)
    
    ext_indices_task1 = extend_task1_indices(markers, random_indices_task1, headers)
    
    ### subtask: mask related nodes, create subgraph
    all_indices = torch.arange(batch_size*23)
    mask = [i for i in all_indices if i not in ext_indices_task1]
    altered_edge_index, _ = subgraph(mask, edge_index=edge_index)
    
    # prepare task 2: predict random node in next graph
    random_indices_task2 = torch.randint(0, 23, (batch_size, 1))
    indices_task2 = markers + random_indices_task2
    task2_true = next_possible_nodes[indices_task2].squeeze(1)
    
    return x, altered_edge_index, task1_true, task2_true, mask, masked_nodes_prim

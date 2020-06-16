import torch
import numpy as np
import pytest

import src.preparation.utils
from src.preparation.datasets import AnomalyDataset, MultiAnomalyDataset
from src.preparation.datasets_helpers import RawData, compute_groups, create_edge_index, transform_time_series_data
from src.preparation.splitting import LeaveOneGroupOutWrapper, TransferLearningWrapper
from src.preparation.transforms import ToTensorTransform

@pytest.fixture
def anomalies():
    return ["bandwidth", "download", "packet_loss"]

@pytest.fixture
def datasets():
    my_datasets = []
    
    anomaly_names = ["bandwidth", "download", "packet_loss"]
    node_names = ["cassandra", "bono", "sprout"]

    for n in node_names:
        my_data = []
        for idx, a in enumerate(anomaly_names):
            for _ in range(3):
                my_data.append(RawData(x=np.random.rand(50,10), y=idx, headers=[], node_name=n, anomaly_name=a))

        dataset = AnomalyDataset(n, anomaly_file_paths={}, ref_header=["blocker"])
        dataset.data = my_data

        my_datasets.append(dataset)        

    return my_datasets 


def test_compute_groups(datasets):
    data = datasets[0].get_data()

    groups = compute_groups(data)
    
    for idx, el in enumerate(groups):
        idx += 1
        if idx % 3 > 0:
            assert el == idx % 3
        else:
            assert el == 3    


def test_create_edge_index():
    a = 10
    b = 10
    edge_index = create_edge_index(a, b)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == (a*b)**2


def test_transform_time_series_data(datasets):
    data = datasets[0].get_data()
    
    data, _ = ToTensorTransform()(data, [])

    graphs, _, _ = transform_time_series_data([data, [], []], window_width=10)
    
    for graph in graphs:
        assert graph.x.shape == (100, 1)
        assert graph.edge_index.shape == (2, 10000)
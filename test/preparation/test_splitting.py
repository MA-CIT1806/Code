import torch
import numpy as np
import pytest

import src.preparation.utils
from src.preparation.datasets import AnomalyDataset, MultiAnomalyDataset
from src.preparation.datasets_helpers import RawData
from src.preparation.splitting import LeaveOneGroupOutWrapper, TransferLearningWrapper


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


def test_logo_splitting(datasets, anomalies):
    logo = LeaveOneGroupOutWrapper(datasets[0])   

    # assert that second group (for validation / testing) has exemplar of each anomaly
    for split in logo.splits:
        for a in anomalies:
            assert any([a == datasets[0].data[idx].anomaly_name for idx in split[1]])


def test_tf_splitting(datasets, anomalies):
    mul_ds = MultiAnomalyDataset(datasets)
    tf = TransferLearningWrapper(mul_ds)

    split = tf.get_split(0, left_out_group="cassandra")

    # assert that second groups consists only of recordings from a specific node
    assert not any(["cassandra" == el.node_name for el in split[0]])
    assert all(["cassandra" == el.node_name for el in split[1]])           
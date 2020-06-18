import os
import re
from functools import reduce

import pandas as pd
import numpy as np
import torch
from src.preparation.datasets import AnomalyDataset


def _get_mappings(data_root_path, exclude_nodes=None, exclude_anomalies=None):
    """Returns a mapping of anomalies to nodes and a mapping of nodes to anomalies."""

    exclude_nodes = [] if exclude_nodes is None else exclude_nodes
    exclude_anomalies = [] if exclude_anomalies is None else exclude_anomalies

    # Get anomalies in anomaly directory
    (_, anomalies, _) = os.walk(data_root_path).__next__()
    # Each anomaly type directory contains directories named after nodes where it was injected
    # Create mapping on anomalies to all nodes it was injected on
    # Exclude all anomalies that should be excluded
    anomaly2nodes = {anomaly: os.walk(os.path.join(data_root_path, anomaly)).__next__()[1] for anomaly in anomalies
                     if anomaly not in exclude_anomalies}
    # Get a list of unique nodes (all nodes where some anomalies were injected)
    # Exclude all nodes that should be excluded
    unique_nodes = set([node for nodes in anomaly2nodes.values()
                        for node in nodes if node not in exclude_nodes])
    # Reverse the mapping. Map the nodes to a list of anomalies which were injected on it. Therefore, check if a node
    # directory is in the respective anomaly directory
    node2anomalies = {node: [anomaly for anomaly in anomaly2nodes if node in anomaly2nodes[anomaly]]
                      for node in unique_nodes}
    return anomaly2nodes, node2anomalies


def _get_file_paths(data_root_path, node2anomalies):
    # Helper function to make dict comprehension look cleaner
    def build_path(anomaly, node):
        return os.path.normpath(os.path.join(os.path.join(data_root_path, anomaly), node))

    def sort_paths(paths):
        paths.sort()
        return paths

    # Create anomaly dataset for each node
    node2filepaths = {}
    for node, anomalies in node2anomalies.items():
        anomaly_file_dirs = {anomaly: build_path(
            anomaly, node) for anomaly in anomalies}
        anomaly_file_paths = {anomaly: [os.path.join(anomaly_file_dir, file_name)
                                        for file_name in sort_paths((list(os.walk(anomaly_file_dir).__next__()[2])))]
                              for anomaly, anomaly_file_dir in anomaly_file_dirs.items()}
        node2filepaths[node] = anomaly_file_paths
    return node2filepaths


# Creates anomaly dataset in which
def read_nodegroup_2_filepaths(data_root_path, exclude_nodes=None, exclude_anomalies=None, group_pattern=""):
    """
    Creates a dictionary of file paths, where the keys are the respective service components,
    and the values are the corresponding file paths, i.e. paths describing the location of data belonging to a node.
    """

    _, node2anomalies = _get_mappings(
        data_root_path, exclude_nodes, exclude_anomalies)
    node2filepaths = _get_file_paths(data_root_path, node2anomalies)

    # Create mapping from node group to file paths. The node group is inferred from the node name by applying the
    # group_pattern
    nodegroup2filepaths = {}
    for node, filepaths in node2filepaths.items():
        node_group = re.sub(group_pattern, '', node)
        if node_group in nodegroup2filepaths:
            nodegroup2filepaths[node_group].update({k: (nodegroup2filepaths[node_group][k] + filepaths[k])
                                                    for k in nodegroup2filepaths[node_group].keys() if k in filepaths})
        else:
            nodegroup2filepaths[node_group] = filepaths

    return nodegroup2filepaths


def load_datasets(node_names, data_path, exclude_anomalies=None):
    """Given a list of service components (nodes), load the data for each node."""

    nodegroup2filepaths = read_nodegroup_2_filepaths(data_path)
    
    datasets = []
    for node_name in node_names:
        anomaly_file_paths = nodegroup2filepaths[node_name]
        datasets.append(AnomalyDataset(node_name, anomaly_file_paths=anomaly_file_paths, exclude_anomalies=exclude_anomalies))
  

    # make sure all datasets have same number of classes
    if len(set([d.num_classes for d in datasets])) != 1:
        raise ValueError("datasets have varying number of classes.")

    return datasets


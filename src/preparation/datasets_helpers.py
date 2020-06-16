import torch
import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from torch_geometric.data import DataLoader, Data

full_anomaly_label_dict = {
    "bandwidth": 0,
    "download": 1,
    "mem_leak": 2,
    "packet_duplication": 3,
    "packet_loss": 4,
    "stress_cpu": 5, 
    "stress_hdd": 6,
    "stress_mem": 7  
}

reduced_anomaly_label_dict = {
       "download": 1,
       "mem_leak": 3,
       "stress_cpu": 2, 
       "stress_hdd": 0,
       "stress_mem": 4  
    }

def compute_groups(data):
    """Given a list of RawData-objects, creates groups in preparation for LOGO cross-validation."""

    unique_labels = list(set([str(d) for d in data]))
    group_val_dict = {label: 1 for label in unique_labels}
    groups = []
    for d in data:
        groups.append(group_val_dict[str(d)])
        group_val_dict[str(d)] += 1

    return groups

class RawData():
    r"""Wrapper for a data recording.

    Args:
        x (np.array, optional): The value array of the recording.
        headers (list, optional): A list of headers, extracted from the original csv-file.
        y (int, optional): label of the recording.
        anomaly_name (string, optional): The name of the corresponding anomaly, e.g. bandwith or download.
        node_name (string, optional): The name of the corresponding service component, e.g. bono or cassandra.
    """
    def __init__(self, x=None, headers=None, y=None, anomaly_name=None, node_name=None):
        
        self.x = x
        self.headers = headers
        self.y = y
        self.anomaly_name = anomaly_name
        self.node_name = node_name
        
    @staticmethod
    def create_from_ref(o, x=None, headers=None, y=None, anomaly_name=None, node_name=None):
        parameters = {
            "x": x if x is not None else o.x,
            "headers": headers or o.headers,
            "y": y if y is not None else o.y,
            "anomaly_name": anomaly_name or o.anomaly_name,
            "node_name": node_name or o.node_name
        }
        
        return RawData(**parameters)
        
    def __str__(self):
        return "y={},anomaly={},node={}".format(self.y, self.anomaly_name, self.node_name)
   

def _check_for_corrupted_data(lists, prefix=""):
    """For a list of torch-tensors, check if elements are either nan or inf."""

    for l in lists:
        for el in l:
            if torch.is_tensor(el.x):
                if torch.isnan(el.x).sum() > 0 or torch.isinf(el.x).sum() > 0:
                    print("Data is partially corrupted after: {}".format(prefix))                
            else:    
                if np.isnan(el.x).sum() > 0 or np.isinf(el.x).sum() > 0:
                    print("Data is partially corrupted after: {}".format(prefix))


def read_data(node_name, anomaly_file_paths, pattern, ref_header, exclude_anomalies=None):
    """Read all data given a dictionary paths, where each entry describes all recordings of an anomaly."""

    data = []
    
    if not exclude_anomalies:
        exclude_anomalies = []
    
    for anomaly, file_paths in anomaly_file_paths.items():
        if anomaly not in exclude_anomalies:
            for file_path in file_paths:
                file_prefix = re.findall(pattern, os.path.basename(file_path))
                if not file_prefix:
                    raise ValueError("Illegal file name {}. Files are expected to have a number as prefix."
                                     .format(file_path))
                if ref_header is not None:
                    csv_data = pd.read_csv(file_path, usecols=ref_header).drop(
                        ["time", "tags"], axis=1)
                else:
                    csv_data = pd.read_csv(file_path).drop(
                        ["time", "tags"], axis=1)
                data.append(RawData(
                    x=csv_data.values, 
                    headers=list(csv_data.columns),
                    y=reduced_anomaly_label_dict[anomaly], 
                    anomaly_name=anomaly, 
                    node_name=node_name))
     
    return data, dict([(value, key) for key, value in reduced_anomaly_label_dict.items()])


def create_edge_index(window_width, num_channels):
    """Create a representation for the connectivity of a fully connected graph."""

    num_nodes = window_width * num_channels
    edge_index = torch.zeros((2, num_nodes*num_nodes), dtype=torch.long)
    idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index[0][idx] = i
            edge_index[1][idx] = j
            idx += 1

    return edge_index


def transform_time_series_data(data_list, window_width=10, sliding_window=10, init_fn=None, extraction_target="window", **kwargs):
    """extract slices of time series data and model them as graphs."""

    flatten_slice = kwargs.get("flatten_slice", True)

    new_data_list = []
    
    edge_indices = {}
    
    def _get_edge_index(sample):
        num_channels = sample.shape[0]
        if edge_indices.get(num_channels, None) is not None:
            return edge_indices.get(num_channels)
        else:
            ww = window_width if flatten_slice else 1
            res = create_edge_index(ww, num_channels).to("cpu")
            
            edge_indices[num_channels] = res
            return res
        

    for data in data_list:
        for d in data:
            d.x = d.x.T if d.x.shape[0] > d.x.shape[1] else d.x 
        inner_data_list = []

        for idx in range(len(data)):
            arr = data[idx].x
            arr_label = data[idx].y
            
            edge_index = _get_edge_index(arr)
            
            temp_data_list = []
                        
            indices = range(0, (arr.size(1) - window_width) + 1, sliding_window)       

            for i in range(len(indices)):
                new_arr = arr[:, indices[i]:indices[i]+window_width]
                new_arr = torch.from_numpy(
                    new_arr).to("cpu") if not torch.is_tensor(new_arr) else new_arr.to("cpu")

                new_arr = new_arr.reshape(-1, 1) if flatten_slice else new_arr
                new_label = arr_label.clone()
               
                temp_data_list.append(Data(x=new_arr,
                                           y=new_label,
                                           headers=data[idx].headers,
                                           edge_index=edge_index, 
                                           num_nodes=new_arr.shape[0]
                                          ))
            
            for i in range(0, len(indices)):
                if i < len(indices) - 1:
                    temp_data_list[i].next_possible_nodes = temp_data_list[i+1].x
                else:
                    temp_data_list[i].next_possible_nodes = temp_data_list[i].x # change in time wont be that big
                    
            if extraction_target == "window":
                inner_data_list += temp_data_list
            
            elif extraction_target == "sequence":
                # unload gpu by splitting sequence (since GRU is utilized if sequence is used)
                bs = math.floor(len(temp_data_list) / 3) if len(temp_data_list) > 200 else len(temp_data_list)
                iterator = iter(DataLoader(temp_data_list, 
                          batch_size=bs,
                          shuffle=True,
                          drop_last=True,
                          worker_init_fn=init_fn,
                          num_workers=0))
                for b in iterator:
                    b = b.to("cpu")  

                    inner_data_list.append({
                        "x": b.x,
                        "y": b.y,
                        "edge_index": b.edge_index,
                        "batch": b.batch,
                        "num_nodes": b.num_nodes
                    })      

        new_data_list.append(inner_data_list)

    result = new_data_list

    return result

###############################################
###############################################
###### below: currently not needed ############
def augmentate(given_input, augmentation_config):
    dists = []
        
    # extract configuration details
    sigma = augmentation_config.get("sigma")
    mu = augmentation_config.get("mu")
    proba = augmentation_config.get("proba")
    cat = augmentation_config.get("cat", False)
    online = augmentation_config.get("online", False)
    offline = augmentation_config.get("offline", False)
    
    # setup distributions
    if isinstance(sigma, (int, float)) and isinstance(mu, (int, float)):
        dists.append(torch.distributions.normal.Normal(mu, sigma))
    elif isinstance(sigma, list) and isinstance(mu, list):
        for i in range(len(sigma)):
            dists.append(torch.distributions.normal.Normal(mu[i], sigma[i]))
                       
    x_new = []
    y_new = [] 
    # augmentate data
    if offline and not online:
        x_train_list, y_train_list = given_input
        for i in range(len(x_train_list)):
                x_new.append(x_train_list[i])
                y_new.append(y_train_list[i])
                if proba >= np.random.random_sample():
                    for dist in dists:
                        noise = dist.sample(x_train_list[i].shape).to(x_train_list[i].device)
                        if cat:
                            x_new.append(x_train_list[i] + noise)
                            y_new.append(y_train_list[i])
                        else:
                            x_new[i] += noise
    
        return x_new, y_new
    
    if online and not offline:
        x_train, batch = given_input.x, given_input.batch
        for label in batch.unique():
            if proba >= np.random.random_sample():
                indices = batch == label
                for dist in dists:
                    noise = dist.sample((indices.sum(), x_train.size(1))).to(x_train.device)
                    x_train[indices] += noise
        return x_train 

    
def visualize_processing_steps(my_list):
    min_x = min([list(el.values())[0].size(0) for el in my_list])
    _, axes = plt.subplots(math.ceil(len(my_list) / 2), 2, figsize=(16,10))
    for idx, l in enumerate(my_list):
        prefix = "" if idx == 0 else "After "
        if axes.ndim == 2:
            x, y = idx // 2, idx % 2
            axes[x, y].plot(range(min_x), list(l.values())[0][:min_x, 0])
            axes[x, y].set_title("{}{}".format(prefix, list(l.keys())[0]))
        else:
            x = idx
            axes[x].plot(range(min_x), list(l.values())[0][:min_x, 0])
            axes[x].set_title("{}{}".format(prefix, list(l.keys())[0]))            
    plt.show()     


class RandomBatchSampler(torch.utils.data.sampler.Sampler):
    
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)
        t_res = torch.arange((n // self.batch_size) * self.batch_size).reshape(-1, self.batch_size)
        t_res = t_res[torch.randperm(t_res.size(0))]
        
        return iter(torch.flatten(t_res).tolist())
        
    def __len__(self):
        return len(self.data_source)
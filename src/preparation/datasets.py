import torch
import pandas as pd
import os
import os.path as osp
import re
import bisect
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from functools import reduce
from src.preparation.utils import *
from src.preparation.datasets_helpers import *
from src.preparation.transforms import TransformCompose
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
import torch_geometric.data as TGD


class MultiAnomalyDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MultiAnomalyDataset, self).__init__(datasets)
        
    def _indices_to_datasets(self, train_indices):
        
        res_dict = {}
        
        for idx in train_indices:
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
            if dataset_idx in res_dict:
                res_dict[dataset_idx].append(sample_idx)
            else:
                res_dict[dataset_idx] = [sample_idx]
                
        return res_dict   
    
    def get_data(self):
        data = []
        for ds in self.datasets:
            data += ds.get_data()
        return data    
    
    def get_groups(self):
        return compute_groups(self.get_data())    
    
    @property
    def num_classes(self):
        return self.datasets[0].num_classes
    
    @property
    def name(self):
        return "[{}]".format("-".join([d.name for d in self.datasets]))
        
    def set_transform(self, train_indices, data_transform_steps={}):
        for key, value in self._indices_to_datasets(train_indices).items():
            self.datasets[key].set_transform(value, data_transform_steps=data_transform_steps) 
            
    def __str__(self):
        return "num_classes={}, num_raw_data_sources={}".format(self.num_classes, len(self.get_data()))            
            

class AnomalyDataset(Dataset):

    pattern = "(^[0-9]*)"

    def __init__(self, node, anomaly_file_paths=None, ref_header=None, exclude_anomalies=None):
        
        if not exclude_anomalies:
            exclude_anomalies = []
        
        if not ref_header:
            headers = self._read_headers(anomaly_file_paths)
            ref_header = self._unify_headers(headers)
        self.ref_header = ref_header
                
        self.name = node
        
        self.transformed = None
                
        self.data, self.class_name_mapping = read_data(node, anomaly_file_paths, self.pattern, ref_header, exclude_anomalies=exclude_anomalies)
        self.num_classes = len(self.class_name_mapping.values())
        
    @staticmethod
    def _read_headers(anomaly_file_paths):
        headers = []
        for file_paths in anomaly_file_paths.values():
            headers = [pd.read_csv(
                file_path, nrows=2).columns.values for file_path in file_paths]
        return headers

    @staticmethod
    def _unify_headers(headers):
        unified_header = reduce(lambda x, y: set(
            x).intersection(set(y)), headers)
        return unified_header
    
    def set_transform(self, train_indices, data_transform_steps={}):
        rest_indices = [i for i in range(len(self.data)) if i not in train_indices]
        
        train_data = [self.data[i] for i in train_indices]
        rest_data = [self.data[i] for i in rest_indices]
        
        transformer = TransformCompose(**data_transform_steps)
        
        train_data_transformed, rest_data_transformed  = transformer(train_data, rest_data)
        
        self.transformed = list(range(len(self.data)))
        
        for idx, i in enumerate(train_indices):
            self.transformed[i] = train_data_transformed[idx]
        for idx, i in enumerate(rest_indices):
            self.transformed[i] = rest_data_transformed[idx]
            
    
    def get_data(self):
        return self.data
    
    def get_groups(self):
        return compute_groups(self.data)
            
    def __getitem__(self, idx):
        if self.transformed is not None:
            return self.transformed[idx]
        return self.data[idx] 
    
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return "num_classes={}, num_raw_data_sources={}".format(self.num_classes, len(self.data))
    
#######################################    
class AnomalySubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        
    def set_transform(self, train_indices, data_transform_steps={}):
        real_train_indices = [self.indices[i] for i in train_indices]
        self.dataset.set_transform(real_train_indices, data_transform_steps=data_transform_steps)
        
    def get_data(self):
        dataset = self.dataset.get_data()
        return [dataset[i] for i in self.indices]
    
    @property
    def name(self):
        return self.dataset.name
    
    @property
    def class_name_mapping(self):
        return self.dataset.class_name_mapping
    
    def get_groups(self):
        return compute_groups(self.get_data())        
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]   
#######################################
    
class GraphDataset():  
    
    def __init__(self, train_data, valid_data=[], test_data=[], graph_dataset_config={}):
        
            self.configuration = {
                "sliding_window": graph_dataset_config.get("graph_dataset_config", 1),
                "window_width": graph_dataset_config.get("window_width", 1),
                "flatten_slice": graph_dataset_config.get("graph_dataset_config", False),
                "complete_batches": graph_dataset_config.get("complete_batches", False),
                "use_custom_sampler": graph_dataset_config.get("use_custom_sampler", False),
                "graph_transform": graph_dataset_config.get("graph_transform", lambda x: x),
                "batch_size": graph_dataset_config.get("batch_size", 32),
                "extraction_target": graph_dataset_config.get("extraction_target", "window"),
                "init_fn": graph_dataset_config.get("_init_fn"),
                "device": graph_dataset_config.get("device"),
                "shuffle_settings": graph_dataset_config.get("shuffle_settings", {
                    "train": True, 
                    "valid": False, 
                    "test": False
                })
            }
            
            data = [train_data, valid_data, test_data]
        
            self.train_data, self.valid_data, self.test_data = transform_time_series_data(data, **self.configuration)
            
            
    @property
    def y_test(self):
        my_list = []
        for d in self.test_data:
            my_list.append(d["y"])
        return torch.cat(my_list, dim=0)
    
    @property
    def batch_size(self):
        return self.configuration["batch_size"]
    
    @property
    def init_fn(self):
        return self.configuration["init_fn"]
    
    def _list_to_loader(self, data_list, shuffle):
        batch_size = None
        data_loader_class = None
        if self.configuration["extraction_target"] == "window":
            batch_size = self.configuration["batch_size"]
            data_loader_class = TGD.DataLoader
        else:
            batch_size = 1
            data_loader_class = DataLoader
        
        return data_loader_class(data_list, 
                              shuffle=shuffle,
                              batch_size=batch_size,
                              worker_init_fn=self.configuration["init_fn"],
                              num_workers=0)        
            
    def get_loaders(self):
        config = self.configuration["shuffle_settings"]
        train_loader = self._list_to_loader(self.train_data, config["train"])
        valid_loader = self._list_to_loader(self.valid_data, config["valid"])
        test_loader = self._list_to_loader(self.test_data, config["test"])
        
        return train_loader, valid_loader, test_loader
                              
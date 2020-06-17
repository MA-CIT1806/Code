import logging

from sklearn.model_selection import LeaveOneGroupOut
from torch.utils.data import Subset

from src.preparation.datasets import AnomalySubset

class LeaveOneGroupOutWrapper:
    """Wrapper for LOGO-cross-validator. The dataset must be an instance of a dataset defined in datasets.py"""
    def __init__(self, dataset):
        self.dataset = dataset

        self.logo = LeaveOneGroupOut()
        self.splits = list(self.logo.split(dataset.get_data(), y=None, groups=dataset.get_groups()))
        self.split_iter = iter(self.splits)

    def get_split(self, index=-1, data_transform_steps={}):
        all_but_one_group_indices, left_out_group_indices = self._get_split_indices(index)
        if all_but_one_group_indices is None or left_out_group_indices is None:
            return None, None
        
        # preprocess all data based on training data
        self.dataset.set_transform(all_but_one_group_indices, data_transform_steps=data_transform_steps)
        
        return AnomalySubset(self.dataset, all_but_one_group_indices), AnomalySubset(self.dataset, left_out_group_indices)

    def _get_split_indices(self, index):
        try:
            if not index or index < 0:
                left_split, right_split = next(self.split_iter)
            else:
                left_split, right_split = self.splits[index]
            print("Train and validation indices: {} \n Test indices: {}".format(left_split, right_split))
        except Exception as e:
            logging.warning("Splitting failed.", exc_info=e)
            return None, None
        return left_split, right_split
    
    
    
class TransferLearningWrapper:
    """
    Wrapper for transfer learning. Does not conduct a LOGO-splitting,
    but excludes a complete node for use as validation set.
    The dataset must be an instance of a dataset defined in datasets.py
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def get_split(self, index=-1, data_transform_steps={}, left_out_group="cassandra"):
        all_but_one_group_indices = []
        left_out_group_indices = []
        
        data_list = self.dataset.get_data()
        
        for i, d in enumerate(data_list):
            if d.node_name == left_out_group:
                left_out_group_indices.append(i)
            else:
                all_but_one_group_indices.append(i)
                
        print("Train and validation indices: {} \n Test indices: {}".format(all_but_one_group_indices, left_out_group_indices)) 
            
        # preprocess all data based on training data
        self.dataset.set_transform(list(range(len(data_list))), data_transform_steps=data_transform_steps)        
        
        return AnomalySubset(self.dataset, all_but_one_group_indices), AnomalySubset(self.dataset, left_out_group_indices)    
import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from src.preparation.datasets_helpers import _check_for_corrupted_data, RawData

class TransformCompose(object):
    """Composes several custom-transforms together.

    Args:
        transforms (list of :obj:`transform` objects): List of transforms to
            compose.
    """
    def __init__(self, transforms=[], clear_after_use=False):
        self.transforms = transforms + [ToTensorTransform()]
        self.clear_after_use = clear_after_use          
            
    def __call__(self, train_list, rest_list):
        
        for t in self.transforms:
            train_list, rest_list = t(train_list, rest_list, clear_after_use=self.clear_after_use)
        return train_list, rest_list

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))
    
    
class BaseTransform(object):
    """Base class for data transforms."""
    def __init__(self):
        self.name = ""
        
    def transform(self, func, train_list, rest_list):
        train_list_transformed = [func(data) for data in train_list]
        rest_list_transformed = [func(data) for data in rest_list]
        
        _check_for_corrupted_data([train_list_transformed, rest_list_transformed], self.name)    
        
        return train_list_transformed, rest_list_transformed

    
class DifferenceTransform(BaseTransform):
    r"""Computes the mean over the kth differences along the time dimension.
    Args:
        k (int, optional): number of differences. (default: :obj:`1`)
    """

    def __init__(self, k=1, append=False):
        super(DifferenceTransform, self).__init__()
        self.name = "Kth-Difference"
        
        self.k = k
        self.append = append

    def __call__(self, train_list, rest_list, **kwargs):
        
        print("Apply Kth-Difference...")
        print("   K={}".format(self.k))
        
        def kth_difference(data):
            x = data.x
            
            my_list = []
            for i in range(self.k):
                idx = i + 1
                diff = x[idx:] - x[:-idx]
                if self.k > 1 and idx < self.k:
                    diff = diff[:-(self.k-idx)]
                my_list.append(diff)
            x = np.stack(my_list).mean(0)
            
            if self.append:
                seq_length = min(x.shape[0], data.x.shape[0])
                x = np.concatenate([x[:seq_length], data.x[:seq_length]], axis=1)
            
            return RawData.create_from_ref(data, x=x)
        
        return self.transform(kth_difference, train_list, rest_list)

    def __repr__(self):
        return '{}(k={})'.format(self.__class__.__name__, self.k)
    

class MinMaxTransform(BaseTransform):
    r"""Normalizes given data based on training data.
    Args:
        target_min (int, optional): lower bound. (default: :obj:`0`)
        target_max (int, optional): upper bound. (default: :obj:`1`)
    """

    def __init__(self, target_min=0, target_max=1):
        super(MinMaxTransform, self).__init__()
        self.name = "Normalization"
        
        self.target_min = target_min
        self.target_max = target_max
        self.arr_min = None
        self.arr_max = None
        self.denom = None

    def __call__(self, train_list, rest_list, clear_after_use=False):
        
        if clear_after_use:
            self.denom = None
        
        if self.denom is None:
            train_stacked = np.concatenate([d.x for d in train_list], axis=0)

            self.arr_max = train_stacked.max(axis=0)
            self.arr_min = train_stacked.min(axis=0)

            denom = self.arr_max - self.arr_min
            denom[denom == 0] = 1  # Prevent division by 0 
            self.denom = denom
            
        print("Apply MinMax...")
        print("   target_min={}, target_max={}".format(self.target_min, self.target_max))            

        def arr_norm(data):
            x = data.x
            
            nom = (x - self.arr_min) * (self.target_max - self.target_min)
            x = self.target_min + nom / self.denom
            
            return RawData.create_from_ref(data, x=x)

        return self.transform(arr_norm, train_list, rest_list)
    
    def __repr__(self):
        return '{}(target_min={:.2f}, target_max={:.2f})'.format(
            self.__class__.__name__, 
            self.target_min, 
            self.target_max)    

    
class StandardTransform(BaseTransform):
    r"""Standardizes given data based on training data, e.g. zero mean and unit variance.
    """

    def __init__(self):
        super(StandardTransform, self).__init__()
        self.name = "Standardization"
        
        self.arr_std = None
        self.arr_mean = None

    def __call__(self, train_list, rest_list, clear_after_use=False):
        
        if clear_after_use:
            self.arr_mean = None
            self.arr_std = None
        
        if self.arr_mean is None and self.arr_std is None:
            train_stacked = np.concatenate([d.x for d in train_list], axis=0)
            
            self.arr_std = train_stacked.std(axis=0)
            self.arr_mean = train_stacked.mean(axis=0)

            self.arr_std[self.arr_std == 0] = 1 # Prevent division by 0
            
        print("Apply Standardization...")

        def arr_stand(data):
            x = data.x
            
            x = (x - self.arr_mean) / self.arr_std
            
            return RawData.create_from_ref(data, x=x)

        return self.transform(arr_stand, train_list, rest_list)
    
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)    

    
class ToTensorTransform(BaseTransform):
    """Transforms data to pytorch-tensors."""
    def __init__(self):
        super(ToTensorTransform, self).__init__()
        self.name = "To-Tensor"
        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')
        
    def __call__(self, train_list, rest_list, **kwargs):
        
        def to_tensor(data):
            x, y = data.x, data.y
            
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(np.array([y])).long().to(self.device)
            
            return RawData.create_from_ref(data, x=x, y=y)
            
        return self.transform(to_tensor, train_list, rest_list)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    

class LogScaleTransform(BaseTransform):
    r"""Takes the logarithm of the given input.

    Args:
        base (int, optional): The base of the logarithm. Default is natural logarithm.
    """

    def __init__(self, base=None):
        super(LogScaleTransform, self).__init__()
        self.name = "Log-Scaling"
        
        log_func = np.log
        if base is not None and isinstance(base, int):
            if base == 2:
                log_func = np.log2
            if base == 10:
                log_func = np.log10
        self.log_func = log_func
        self.base = base

    def __call__(self, train_list, rest_list, **kwargs):
        
        print("Apply Log-Scaling...")   
        print("   base={}".format("ln" if self.base is None else self.base))
        
        def log_scale(data):
            x = data.x
            
            x = self.log_func(x + 1) # Prevent negative infinity by adding 1
            
            return RawData.create_from_ref(data, x=x)

        return self.transform(log_scale, train_list, rest_list)

    def __repr__(self):
        return '{}(base={})'.format(self.__class__.__name__, self.base)
    
    
class WhiteningTransform(BaseTransform):
    r"""Applies ZCA-Whitening on all data based on training data.

    Args:
        eps (float, optional): Value to add to eigenvalues before taking the square root. (default: :obj:`10**-5`)
    """

    def __init__(self, eps=10**-5):
        super(WhiteningTransform, self).__init__()
        self.name = "Whitening"
        
        self.eps = eps
        self.sigma_neg_sqrt = None
        self.shrinkage_parameter = None

    def __call__(self, train_list, rest_list, clear_after_use=False):
        
        print("Apply Whitening...")
        
        if clear_after_use:
            self.sigma_neg_sqrt = None
            self.shrinkage_parameter = None
        
        if self.sigma_neg_sqrt is None:
            train_stacked = np.concatenate([d.x for d in train_list], axis=0)    
            # Fit LedoitWolf for covariance estimation
            lw = LedoitWolf().fit(train_stacked)
            self.shrinkage_parameter = lw.shrinkage_
            print("   Estimated shrinkage-parameter={:.3f}".format(self.shrinkage_parameter))
            # estimated covariance matrix
            sigma = lw.covariance_
            # eigenvalue decomposition
            eig_values, eig_vectors = np.linalg.eig(sigma)
            # negative square root of eigenvalues
            eig_values_neg_sqrt = np.diag(1 / np.sqrt(eig_values + self.eps))
            # negative square root of sigma
            self.sigma_neg_sqrt = np.dot(np.dot(eig_vectors, eig_values_neg_sqrt), eig_vectors.T)
        
        def tensor_whiten(data):
            x = data.x
            
            x = np.dot(x, self.sigma_neg_sqrt)
            
            return RawData.create_from_ref(data, x=x)

        return self.transform(tensor_whiten, train_list, rest_list)

    def __repr__(self):
        return '{}(eps={:.3f}, shrinkage-parameter={:.3f})'.format(self.__class__.__name__, self.eps, self.shrinkage_parameter)    
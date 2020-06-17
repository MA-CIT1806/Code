import dill
import os
import json
import inspect

def load_pickle(path: str):
    if not os.path.isfile(path):
        raise ValueError("'path' does not point to a file.")

    with open(path, "rb") as file:
        return dill.load(file)


def save_pickle(obj, path: str):
    with open(path, "wb") as file:
        return dill.dump(obj, file)


def create_dir(path_to_dir):
    try:
        os.makedirs(path_to_dir, exist_ok=True)
    except FileExistsError:
        return None
    return path_to_dir


def rename_dir(old, new):
    try:
        os.rename(old, new)
    except Exception as e:
        print(e)
        return None
    return new


def _filter_dictionary(old_dict):
    """Filters a dictionary, such that the filtered dictionary can be converted to a string."""

    new_dict = dict()
    for (key, value) in old_dict.items():
        if isinstance(value, (int, float, str, bool, tuple, list)) or value is None:
            new_dict[key] = str(value)
        elif isinstance(value, dict):
            new_dict[key] = _filter_dictionary(value)
        elif key == "model_class":
            new_dict[key] = str(inspect.getsource(value))
        else:
            new_dict[key] = str(value)

    return new_dict


def store_results(acc, loss, dataset_name, model_name, configs, suffix=""):
    """Simple way of storing results. Arguments are (recursively) converted to strings and written to file."""

    path = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    file_name = "acc_{:.3f}_loss_{:.3f}_{}_{}".format(acc, loss, dataset_name, model_name)
    with open(os.path.join(path, '{}{}.txt'.format(file_name, suffix)), 'w') as file:
        for config in configs:
            file.write(json.dumps(_filter_dictionary(config)))


def setup_tb_log_dir(datetime_string, dataset_name, model_name):
    """Creates a subfolder in the tensorboard-log directory for a concrete experiment."""

    root_path = _get_tb_base_log_dir()
    if root_path is not None:
        return os.path.join(root_path, "{}-{}-{}".format(
            datetime_string,
            dataset_name,
            model_name))


def _get_tb_base_log_dir():
    """Return the base directory for tensorboard-logs. Will create directory if it does not exist yet."""

    root_path = os.path.join(os.path.dirname(__file__), "..", "..", "tb_logs")
    return create_dir(root_path)


def setup_seed(seed=123):
    """If required, sets a seed in all necessary places. This will favor reproducible results."""

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    def _init_fn():
        np.random.seed(seed)

    print("pytorch-version: {}".format(torch.__version__))
    return _init_fn


def setup_device(local_path, remote_path):
    """Setup the respective device, choose the right path to the data and set certain cuda flags."""

    import torch
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    device = torch.device("cpu")
    data_path = local_path
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        data_path = remote_path
        if torch.cuda.device_count() == 2:
            torch.cuda.set_device(1)
        torch.cuda.empty_cache()
        print("Run Script with Cuda on: {} , Device: {}".format(
            torch.cuda.get_device_name(), torch.cuda.current_device()))

    return device, data_path

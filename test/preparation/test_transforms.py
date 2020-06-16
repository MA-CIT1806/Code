import torch
import numpy as np
import pytest

import src.preparation.transforms as T
from src.preparation.datasets_helpers import RawData


@pytest.fixture
def rest_list():
    return []


@pytest.fixture
def train_list():
    arr = np.random.rand(200, 10)
    return [RawData(x=arr, y=1)]


def test_norm_transform(train_list, rest_list):

    t = T.MinMaxTransform()

    t_list, r_list = t(train_list, rest_list)

    assert len(r_list) == 0
    assert len(t_list) > 0

    for o in t_list:
        assert np.allclose(np.amax(o.x, axis=0), np.ones(o.x.shape[1]))
        assert np.allclose(np.amin(o.x, axis=0), np.zeros(o.x.shape[1]))


def test_std_transform(train_list, rest_list):

    t = T.StandardTransform()

    t_list, r_list = t(train_list, rest_list)

    assert len(r_list) == 0
    assert len(t_list) > 0

    for o in t_list:
        assert np.allclose(np.mean(o.x, axis=0), np.zeros(o.x.shape[1]))
        assert np.allclose(np.std(o.x, axis=0), np.ones(o.x.shape[1]))


def test_log_transform(train_list, rest_list):

    t = T.LogScaleTransform()

    t_list, r_list = t(train_list, rest_list)

    assert len(r_list) == 0
    assert len(t_list) > 0

    for idx, o in enumerate(t_list):
        assert np.allclose(o.x, np.log(train_list[idx].x + 1))


def test_whitening_transform(train_list, rest_list):

    t = T.WhiteningTransform()

    t_list, r_list = t(train_list, rest_list)

    assert len(r_list) == 0
    assert len(t_list) > 0

    for o in t_list:
        assert np.allclose(np.diag(np.cov(o.x.T)),
                           np.ones(o.x.shape[1]), rtol=1.5e-01)


def test_tensor_transform(train_list, rest_list):

    t = T.ToTensorTransform()

    t_list, r_list = t(train_list, rest_list)

    assert len(r_list) == 0
    assert len(t_list) > 0

    for o in t_list:
        assert torch.is_tensor(o.x)

import pytest
import torch
from src.modeling.utils import chebyshev_polynomials

def test_chebyshev_polynomials():

    base = torch.rand(10,10)

    polynomials = chebyshev_polynomials(base, 3)

    assert len(polynomials) == 3

    assert torch.allclose(polynomials[0], torch.ones_like(base))
    assert torch.allclose(polynomials[1], base)
    assert torch.allclose(polynomials[2], 2*base*base - torch.ones_like(base))

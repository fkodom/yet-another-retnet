from typing import Optional

import pytest
import torch
from torch import Tensor

from yet_another_retnet.retention import (
    MultiScaleRetention,
    retention_chunkwise,
    retention_parallel,
    retention_recurrent,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
# Set deterministic CUDA ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_retention_parallel_forward():
    # TODO
    pass


def test_retention_recurrent_forward():
    # TODO
    pass


def test_retention_chunkwise_forward():
    # TODO
    pass


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("seq_length", [16])
@pytest.mark.parametrize("hidden_dim", [4, 8])
@pytest.mark.parametrize("chunk_size", [1, 4, 16])
def test_equivalent_retention_formulations(
    batch_size: int,
    num_heads: int,
    seq_length: int,
    hidden_dim: int,
    chunk_size: int,
):
    size = (batch_size, num_heads, seq_length, hidden_dim)
    query = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    key = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    value = torch.randn(*size, device=DEVICE, dtype=DTYPE)

    # Parallel formulation
    y_parallel, _ = retention_parallel(query, key, value)

    # Recurrent formulation
    y_recurrent = torch.zeros_like(y_parallel)
    prev_state: Optional[Tensor] = None
    for i in range(seq_length):
        q, k, v = query[:, :, i], key[:, :, i], value[:, :, i]
        y_recurrent[:, :, i], prev_state = retention_recurrent(q, k, v, prev_state)

    torch.testing.assert_close(y_parallel, y_recurrent, rtol=1e-4, atol=1e-4)
    recurrent_state = prev_state

    # Chunkwise formulation
    y_chunkwise = torch.zeros_like(y_parallel)
    prev_state = None
    for i in range(0, seq_length, chunk_size):
        q = query[:, :, i : i + chunk_size]
        k = key[:, :, i : i + chunk_size]
        v = value[:, :, i : i + chunk_size]
        y_chunkwise[:, :, i : i + chunk_size], prev_state = retention_chunkwise(
            q, k, v, prev_state
        )

    torch.testing.assert_close(y_parallel, y_chunkwise, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(recurrent_state, prev_state, rtol=1e-4, atol=1e-4)


def test_multiscale_retention_forward_parallel():
    # TODO
    pass


def test_multiscale_retention_forward_recurrent():
    # TODO
    pass


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("seq_length", [8])
@pytest.mark.parametrize("embed_dim", [16, 32])
@pytest.mark.parametrize("chunk_size", [1, 4, 16])
def test_equivalent_multiscale_formulations(
    batch_size: int,
    num_heads: int,
    seq_length: int,
    embed_dim: int,
    chunk_size: int,
):
    size = (batch_size, seq_length, embed_dim)
    query = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    key = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    value = torch.randn(*size, device=DEVICE, dtype=DTYPE)
    mhr = MultiScaleRetention(embed_dim, num_heads, device=DEVICE, dtype=DTYPE).eval()

    y_parallel, _ = mhr.forward_parallel(query, key, value)

    y_recurrent = torch.zeros_like(y_parallel)
    prev_state: Optional[Tensor] = None
    for i in range(seq_length):
        q, k, v = query[:, i], key[:, i], value[:, i]
        y_recurrent[:, i], prev_state = mhr.forward_recurrent(
            q, k, v, seq_idx=i, prev_state=prev_state
        )
    recurrent_state = prev_state

    torch.testing.assert_close(y_parallel, y_recurrent, rtol=1e-4, atol=1e-4)

    y_chunkwise = torch.zeros_like(y_parallel)
    prev_state = None
    for i in range(0, seq_length, chunk_size):
        q = query[:, i : i + chunk_size]
        k = key[:, i : i + chunk_size]
        v = value[:, i : i + chunk_size]
        y_chunkwise[:, i : i + chunk_size], prev_state = mhr.forward_chunkwise(
            q, k, v, start_idx=i, prev_state=prev_state
        )

    torch.testing.assert_close(y_parallel, y_chunkwise, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(recurrent_state, prev_state, rtol=1e-4, atol=1e-4)

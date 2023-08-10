from typing import Optional, Sequence

import pytest
import torch
from torch import Tensor

from yet_another_retnet.retnet import RetNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
# Set deterministic CUDA ops
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_retnet_forward_parallel():
    # TODO
    pass


def test_retnet_forward_recurrent():
    # TODO
    pass


@torch.no_grad()
@pytest.mark.parametrize("num_tokens", [10, 100])
@pytest.mark.parametrize("d_model", [16, 32])
@pytest.mark.parametrize("nhead", [1, 2])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_length", [8])
@pytest.mark.parametrize("chunk_size", [1, 4, 8])
def test_retnet_equivalent_formulations(
    num_tokens: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    batch_size: int,
    seq_length: int,
    chunk_size: int,
):
    size = (batch_size, seq_length)
    x = torch.randint(0, num_tokens, size=size, device=DEVICE)
    net = RetNet(
        num_tokens=num_tokens,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        device=DEVICE,
        dtype=DTYPE,
    ).eval()

    y_parallel = net.forward_parallel(x)

    y_recurrent = torch.zeros_like(y_parallel)
    prev_states: Sequence[Optional[Tensor]] = [None] * num_layers
    for i in range(seq_length):
        xr = x[:, i]
        y_recurrent[:, i], prev_states = net.forward_recurrent(
            xr, seq_idx=i, prev_states=prev_states
        )
    recurrent_states = prev_states

    torch.testing.assert_close(y_parallel, y_recurrent, rtol=1e-4, atol=1e-4)

    y_chunkwise = torch.zeros_like(y_parallel)
    prev_states = [None] * num_layers
    for i in range(0, seq_length, chunk_size):
        xc = x[:, i : i + chunk_size]
        y_chunkwise[:, i : i + chunk_size], prev_states = net.forward_chunkwise(
            xc, start_idx=i, prev_states=prev_states
        )

    torch.testing.assert_close(y_parallel, y_chunkwise, rtol=1e-4, atol=1e-4)
    assert len(recurrent_states) == len(prev_states)
    for rstate, cstate in zip(recurrent_states, prev_states):
        torch.testing.assert_close(rstate, cstate, rtol=1e-4, atol=1e-4)

from typing import Optional, Tuple

import torch
from einops import einsum, rearrange
from torch import nn, Tensor


def _build_decay_mask(
    query_length: int,
    key_length: int,
    gamma: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    query_pos = torch.arange(query_length, device=device, dtype=dtype)
    key_pos = torch.arange(key_length, device=device, dtype=dtype)
    distance = torch.abs(query_pos.unsqueeze(-1) - key_pos.unsqueeze(0))
    # Only keep the lower triangular part of the matrix, so that only *past* keys
    # can influence each current query.
    return (gamma**distance).tril_()


def retention_parallel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    group_norm: Optional[nn.GroupNorm] = None,
    decay_gamma: float = 0.9,
) -> Tensor:
    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: hidden_dim
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        gamma=decay_gamma,
        device=query.device,
        dtype=query.dtype,
    )

    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "n s -> () () n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    if group_norm is not None:
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) h d")
        retention = group_norm(retention)
        retention = rearrange(retention, "(b n) h d -> b h n d", b=batch_size)

    return retention


def retention_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    group_norm: Optional[nn.GroupNorm] = None,
    decay_gamma: float = 0.9,
) -> Tuple[Tensor, Tensor]:
    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - d: hidden_dim
    state = einsum(key, value, "b h d, b h m -> b h d m")
    if prev_state is not None:
        state = state + decay_gamma * prev_state
    retention = einsum(query, state, "b h d, b h d m -> b h m")

    if group_norm is not None:
        retention = group_norm(retention)

    return retention, state


if __name__ == "__main__":
    query = torch.randn(1, 2, 8, 4, device="cuda", dtype=torch.float32)
    key = torch.randn(1, 2, 8, 4, device="cuda", dtype=torch.float32)
    value = torch.randn(1, 2, 8, 4, device="cuda", dtype=torch.float32)
    # group_norm = None
    group_norm = nn.GroupNorm(
        2, 4, eps=1e-6, affine=False, device="cuda", dtype=torch.float32
    ).eval()

    with torch.no_grad():
        yp = retention_parallel(query, key, value, group_norm=group_norm)
        print(yp.shape)
        print(yp[:, :, 2])

        prev_state: Optional[Tensor] = None
        for i in range(3):
            q, k, v = query[:, :, i], key[:, :, i], value[:, :, i]
            yr, prev_state = retention_recurrent(
                q, k, v, prev_state, group_norm=group_norm
            )
            print(yr.shape)
            print(yr)

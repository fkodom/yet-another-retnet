from math import log
from typing import Optional, Tuple

import torch
from einops import einsum, rearrange
from torch import nn, Tensor


def _build_decay_gammas(
    num_heads: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    # return 0.9 * torch.ones(num_heads, device=device, dtype=dtype)
    xmin, xmax = log(1 / 32), log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - torch.exp(x)


def _build_decay_mask(
    query_length: int,
    key_length: int,
    decay_gammas: Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    query_pos = torch.arange(query_length, device=device, dtype=dtype)
    key_pos = torch.arange(key_length, device=device, dtype=dtype)

    distance = torch.abs(query_pos.unsqueeze(-1) - key_pos.unsqueeze(0))
    # Set the upper-triangular distances to infinity, so that only *past* keys
    # can affect the current query.  (Setting distance to infinity ensures that
    # the decay matrix is 0 for those positions, since x^(inf) = 0 when -1 < x < 1.
    distance_mask = torch.ones_like(distance, dtype=torch.bool).triu_(diagonal=1)
    distance = distance.masked_fill(distance_mask, float("inf"))

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas**distance


def retention_parallel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    group_norm: Optional[nn.GroupNorm] = None,
    decay_gammas: Optional[Tensor] = None,
) -> Tensor:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1],
            device=query.device,
            dtype=query.dtype,
        )
    decay_mask = _build_decay_mask(
        query_length=query.shape[2],
        key_length=key.shape[2],
        decay_gammas=decay_gammas,
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: hidden_dim
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
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
    decay_gammas: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if decay_gammas is None:
        decay_gammas = _build_decay_gammas(
            num_heads=query.shape[1],
            device=query.device,
            dtype=query.dtype,
        )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - d: hidden_dim
    state = einsum(key, value, "b h d, b h m -> b h d m")
    if prev_state is not None:
        state = state + prev_state * rearrange(decay_gammas, "h -> () h () ()")
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

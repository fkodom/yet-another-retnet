from functools import lru_cache
from math import log
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor, nn

DEFAULT_DEVICE = torch.device("cpu")
ActivationString = Literal["swish", "gelu", "relu"]


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """Return an activation function given a string"""
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    else:
        raise RuntimeError(
            f"Unsupported activation string '{activation}'. "
            "Supported: 'swish', 'gelu', 'relu'"
        )


@lru_cache(maxsize=1)
def _build_decay_gammas(
    num_heads: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Decay values are different for each retention head, following the prescribed
    method in the paper.  Conceptually, I think of each head having a different
    "retention window", which is the effective number of steps back in time that
    the head can attend to.  Retention windows are effectively determined by
    these decay coefficients.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 3.1 (Setup)
    """
    xmin, xmax = log(1 / 32), log(1 / 512)
    x = torch.linspace(xmin, xmax, steps=num_heads, device=device, dtype=dtype)
    return 1 - x.exp_()


@lru_cache(maxsize=1)
def _build_decay_mask(
    num_heads: int,
    query_length: int,
    key_length: int,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """The decay mask is one of the key components that makes *parallel* retention
    equivalent to *recurrent* retention.  The decay coefficients are pre-computed
    and applied to the similarity matrix at once, rather than being applied to
    each element in the recurrent formulation.

    See: https://arxiv.org/pdf/2307.08621v3.pdf, Equation 5
    """
    decay_gammas = _build_decay_gammas(num_heads=num_heads, device=device, dtype=dtype)

    query_pos = torch.arange(query_length, device=device, dtype=dtype).unsqueeze_(-1)
    key_pos = torch.arange(key_length, device=device, dtype=dtype).unsqueeze_(0)
    distance = torch.abs(query_pos - key_pos)
    # Set the upper-triangular distances to infinity, so that only *past* keys
    # can affect the current query.  (Setting distance to infinity ensures that
    # the decay matrix is 0 for those positions, since x^(inf) = 0 when -1 < x < 1.
    distance_mask = torch.ones_like(distance, dtype=torch.bool).triu_(diagonal=1)
    distance = distance.masked_fill_(distance_mask, float("inf"))

    distance = rearrange(distance, "n s -> () n s")
    decay_gammas = rearrange(decay_gammas, "h -> h () ()")
    return decay_gammas**distance


def _build_position_thetas(
    head_dim: int,
    scale: float = 10000,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Positional thetas are different for each value along head_dim, following the
    prescribed method in the paper.  These are used to update the positional
    embeddings in both the parallel and recurrent formulations of retention.
    See: https://arxiv.org/pdf/2307.08621v3.pdf, Section 2.1 (Retention)

    NOTE: The actual values for thetas are not specified in the paper, so I
    copied these values from the official implementation.
    See: https://github.com/microsoft/torchscale/blob/7d231743f4f96c460b7cf0aa0cf242bb192b34f8/torchscale/architecture/retnet.py#L27C1-L28C59
    """
    x = torch.linspace(0, 1, steps=head_dim // 2, device=device, dtype=dtype)
    thetas = 1 / (scale**x)
    return repeat(thetas, "d -> (d n)", n=2)


# NOTE: For the purposes of positional embeddings, we view query/key Tensors as
# complex-valued, where the even-numbered indices are the real part, and the
# odd-numbered indices are the imaginary part.  This makes it easy to compute
# complex values without *actually* using complex dtypes in PyTorch.
# (Complex dtypes have limited support compared to real dtypes.)
#
# I don't re-explain this in the functions below, but it's important to keep in
# mind when reading the code.


def _multiply_by_i(x: Tensor) -> Tensor:
    """Multiply a complex-valued tensor by the imaginary unit 'i'."""
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(start_dim=-2)


@torch.jit.script
def _theta_shift(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # TODO: Add docstring
    return (x * cos) + (_multiply_by_i(x) * sin)


def retention_parallel(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    need_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    decay_mask = _build_decay_mask(
        num_heads=query.shape[1],
        query_length=query.shape[2],
        key_length=key.shape[2],
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    if need_weights:
        return retention, similarity
    else:
        return retention, None


def retention_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    decay_gammas = _build_decay_gammas(
        num_heads=query.shape[1], device=query.device, dtype=query.dtype
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - d: hidden_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    state = einsum(key, value, "b h d, b h m -> b h d m")
    if prev_state is not None:
        state = state + prev_state * rearrange(decay_gammas, "h -> () h () ()")
    retention = einsum(query, state, "b h d, b h d m -> b h m")

    return retention, state


def retention_chunkwise(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    prev_state: Optional[Tensor],
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    decay_gammas = _build_decay_gammas(
        num_heads=query.shape[1], device=query.device, dtype=query.dtype
    )
    decay_mask = _build_decay_mask(
        num_heads=query.shape[1],
        query_length=query.shape[2],
        key_length=key.shape[2],
        device=query.device,
        dtype=query.dtype,
    )

    # einstein notation:
    # - b: batch_size
    # - h: num_heads
    # - n / s: seq_length
    # - d: head_dim
    if scale is None:
        scale = key.size(-1) ** 0.5
    key = key / scale

    # intra-chunk (same as parallel retention)
    similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
    similarity = similarity * rearrange(decay_mask, "h n s -> () h n s")
    retention = einsum(similarity, value, "b h n s, b h s d -> b h n d")

    # cross-chunk (derived from recurrent retention)
    decay_gammas = rearrange(decay_gammas, "h -> () h () ()")
    inner_pos = rearrange(
        torch.arange(key.size(2), device=key.device, dtype=key.dtype) + 1,
        "n -> () () n ()",
    )
    states = einsum(key, value, "b h n d1, b h n d2 -> b h n d1 d2")
    state_decays = decay_gammas ** (key.size(2) - inner_pos)
    state = einsum(states, state_decays, "b h n d1 d2, _ h n _ -> b h d1 d2")
    if prev_state is not None:
        # Update internal state to return to the user
        chunk_decay = decay_gammas ** key.size(2)
        state = state + prev_state * chunk_decay
        # Update the retention Tensor, based on cross-chunk information
        inner_decay = decay_gammas**inner_pos
        retention = retention + (
            einsum(query, prev_state, "b h n d1, b h d1 d2 -> b h n d2") * inner_decay
        )

    return retention, state


class MultiScaleRetention(nn.Module):
    """Multi-scale retention (MSR) layer.  Intended to be (mostly) a drop-in replacement
    for nn.MultiheadAttention, but with the option to use either the parallel or
    recurrent formulation of retention. (Attention only has the parallel formulation.)

    NOTE: As presented in the paper, Multi-Scale Retention includes an explicit
    position embedding, which is based on xPos.  IMO, this is unnecessary and overly
    specific to language modeling, since other domains (e.g. computer vision,
    heterogeneous graphs) will have different positional semantics.

    I have made the relational position embedding optional, so that this module
    can (in theory) support more modalities. Setting 'relative_position=False' will
    remove the positional embedding, and instead rely on the query and key
    embeddings to encode positional information ahead of time (if needed at all).
    See: https://github.com/microsoft/torchscale/issues/48

    Reference:
        "Retentive Network: A Successor to Transformer for Large Language Models"
        https://arxiv.org/pdf/2307.08621v3.pdf
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        relative_position: bool = True,
        bias: bool = True,
        batch_first: bool = True,
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
        group_norm_eps: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        # TODO???
        # add_bias_kv=False,
        # add_zero_attn=False,
        # kdim=None,
        # vdim=None,
    ):
        if not batch_first:
            raise NotImplementedError("batch_first=False is not yet supported")
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.relative_position = relative_position
        self.bias = bias
        self.activation = activation

        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        head_dim = embed_dim // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )

        # The q/k/v projection layers are the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.group_norm = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=embed_dim,
            affine=False,
            eps=group_norm_eps,
            device=device,
            dtype=dtype,
        )
        # The output project is slightly different, due to the gated "swish" layer.
        self.g_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # 'thetas' parameter for updating the relative position embeddings.
        thetas: Optional[Tensor] = None
        if relative_position:
            thetas = _build_position_thetas(
                head_dim=head_dim, device=device, dtype=dtype
            )
        self.thetas: Optional[Tensor]
        self.register_buffer("thetas", thetas)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Double-check that we're following the same initialization as in
        # the paper.  This is a generic initialization for MHA linear layers.
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_normal_(self.v_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)
        nn.init.xavier_normal_(self.g_proj.weight)
        if self.g_proj.bias is not None:
            nn.init.constant_(self.g_proj.bias, 0)

    def forward_parallel(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            indices = torch.arange(q.size(2), device=q.device, dtype=q.dtype)
            indices = rearrange(indices, "n -> () () n ()")
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = indices * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)

            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, weights = retention_parallel(q, k, v, need_weights=need_weights)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, weights

    def forward_recurrent(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        seq_idx: int,
        prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # h - number of heads
        # d - embedding dimension
        #
        # input shape: (b, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.
        q = rearrange(q, "b (h d) -> b h d", h=self.num_heads)
        k = rearrange(k, "b (h d) -> b h d", h=self.num_heads)
        v = rearrange(v, "b (h d) -> b h d", h=self.num_heads)

        if self.relative_position:
            assert self.thetas is not None
            thetas = rearrange(self.thetas, "d -> () () d")
            angles = seq_idx * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)

            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_recurrent(q, k, v, prev_state=prev_state)
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        # Fold heads back into the embedding dimension.
        retention = rearrange(retention, "b h d -> b (h d)")
        retention = self.group_norm(retention)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward_chunkwise(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        start_idx: int,
        prev_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # einstein notation:
        # b - batch size
        # n - sequence length
        # h - number of heads
        # d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate retention heads.  Move the head
        # dimension to position 1 (makes matrix ops *much* faster).
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        if self.relative_position:
            # global (cross-chunk) + intra-chunk relative position embedding
            assert self.thetas is not None
            indices = torch.arange(start_idx, start_idx + q.size(2), device=q.device, dtype=q.dtype)
            indices = rearrange(indices, "n -> () () n ()")
            thetas = rearrange(self.thetas, "d -> () () () d")
            angles = indices * thetas
            sin = torch.sin(angles)
            cos = torch.cos(angles)
            q = _theta_shift(q, sin, cos)
            k = _theta_shift(k, sin, cos)

        # Apply retention then group norm.
        retention, state = retention_chunkwise(q, k, v, prev_state=prev_state)
        # To apply group norm in an equivalent way to the recurrent formulation,
        # we fold the sequence dimension into the batch dimension.  Otherwise,
        # normalization would be applied over the entire input sequence.
        batch_size = retention.size(0)
        retention = rearrange(retention, "b h n d -> (b n) (h d)")
        retention = F.dropout(retention, p=self.dropout, training=self.training)
        retention = self.group_norm(retention)
        # Unfold 'n' from the batch dimension, and fold 'h' back into the embed dim.
        retention = rearrange(retention, "(b n) e -> b n e", b=batch_size)

        # NOTE: Unlike multihead attention, the retention paper applies a "swish"
        # gate to increase the non-linear capacity of the model.  (IMO this is likely
        # to make up for the lack of "softmax" activation in the retention mechanism.)
        #
        # The paper describes the gate as:
        #   g = swish(X * W_g)
        # where X is the input to the layer.  The authors use Retention in a
        # Decoder-only model, the q/k/v inputs are the same (i.e. X = q = k = v).
        # So, I assume that 'query' can equivalently be used as the input.
        gate = self.activation(self.g_proj(query))
        retention = self.out_proj(retention * gate)

        return retention, state

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.forward_parallel(query, key, value, need_weights=need_weights)


if __name__ == "__main__":
    batch_size = 1
    seq_len = 3
    embed_dim = 16
    num_heads = 2

    layer = MultiScaleRetention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    y_chunkwise, chunkwise_state = retention_chunkwise(x, x, x, prev_state=None)
    print(chunkwise_state)
    print("-" * 40)

    y_recurrent = torch.zeros_like(y_chunkwise)
    recurrent_state = None
    for i in range(seq_len):
        y_recurrent[:, :, i, :], recurrent_state = retention_recurrent(
            x[:, :, i, :], x[:, :, i, :], x[:, :, i, :], prev_state=recurrent_state
        )
    print(recurrent_state)

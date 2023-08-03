from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from yet_another_retnet.retention import (
    ActivationString,
    MultiScaleRetention,
    _get_activation_fn,
)


class RetNetDecoderLayer(nn.Module):
    # NOTE: Mostly pulled from 'nn.TransformerDecoderLayer', but with changes:
    #   - use MultiScaleRetention instead of MultiheadAttention
    #   - no cross-attention layer, since retention doesn't play well with that

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[ActivationString, Callable[[Tensor], Tensor]] = "swish",
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        # retention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.retention = MultiScaleRetention(  # type: ignore
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        # feedforward block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype)

        self._reset_parameters()

    def _reset_parameters(self):
        # TODO: Check that we're following the same initialization as the paper
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def _feedforward_block(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

    def forward_parallel(self, x: Tensor) -> Tensor:
        def _retention_block(x: Tensor) -> Tensor:
            x, _ = self.retention.forward_parallel(x, x, x)
            return self.dropout(x)

        if self.norm_first:
            x = x + _retention_block(self.norm1(x))
            x = x + self._feedforward_block(self.norm2(x))
        else:
            x = x + self.norm1(_retention_block(x))
            x = x + self.norm2(self._feedforward_block(x))

        return x

    def forward_recurrent(
        self, x: Tensor, seq_idx: int, prev_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        def _retention_block(x: Tensor) -> Tuple[Tensor, Tensor]:
            x, state = self.retention.forward_recurrent(
                x, x, x, seq_idx=seq_idx, prev_state=prev_state
            )
            return self.dropout(x), state

        # retention block
        if self.norm_first:
            y, state = _retention_block(self.norm1(x))
            x = x + y
            x = x + self._feedforward_block(self.norm2(x))
        else:
            y, state = _retention_block(x)
            x = x + self.norm1(y)
            x = x + self.norm2(self._feedforward_block(x))

        return x, state

    # TODO
    # def forward_chunkwise

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)


if __name__ == "__main__":
    batch_size = 1
    seq_len = 8
    embed_dim = 16
    num_heads = 2
    device = "cuda"
    dtype = torch.float32

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    layer = RetNetDecoderLayer(
        embed_dim, num_heads, dropout=0, dim_feedforward=32, device=device, dtype=dtype
    ).eval()

    with torch.no_grad():
        yp = layer.forward_parallel(x)
        print(yp[:, 2])

        prev_state: Optional[Tensor] = None
        for i in range(3):
            xx = x[:, i]
            yr, prev_state = layer.forward_recurrent(xx, i, prev_state)
            print(yr)

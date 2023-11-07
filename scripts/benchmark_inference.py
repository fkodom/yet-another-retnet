import os
from typing import List, Optional, Sequence, Tuple, Union

import plotly.graph_objects as go
import torch
from torch import Tensor, nn

from yet_another_retnet.retnet import RetNet, retnet_1_3b
from yet_another_retnet.utils.benchmark import benchmark
from yet_another_retnet.utils.profile import profile

NUM_TOKENS = 10000
BATCH_SIZE = 4
SEQ_LENGTHS = [2048, 3072, 4096, 5120, 6144, 7168, 8192]
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32


class TransformerLM(nn.Module):
    def __init__(
        self,
        num_tokens: int,  # usually obtained from the tokenizer
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_batch_size: int = BATCH_SIZE,
        max_seq_length: int = 8192,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_tokens, d_model, device=device, dtype=dtype)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers
        )
        self.out = nn.Linear(d_model, num_tokens, device=device, dtype=dtype)

        # TODO: This may not be identical to what was benchmarked in the paper.
        # They specifically mention a KV cache, and this isn't technically that.
        # (The key/value projections aren't being cached, just the memory values
        # before KV projection.)  However, this seemed easier to implement, since
        # I don't have to fiddle with Flash Attention or custom Transformer layer
        # implementations.  If time allows, consider implementing a KV cache.
        #
        # a rough-and-dirty memory (KV) cache
        self.cache = torch.zeros(
            (max_batch_size, max_seq_length, d_model),
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor, start_pos: int) -> Tensor:
        batch_size, seq_len = x.shape
        x = self.embeddings(x)

        # memory cache
        self.cache[:batch_size, start_pos : start_pos + seq_len] = x
        memory = self.cache[:batch_size, : start_pos + seq_len]
        x = self.decoder.forward(x, memory)
        return self.out(x)


def transformer_1_3b(
    num_tokens: int,  # usually obtained from the tokenizer
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> TransformerLM:
    """Transformer configuration to match RetNet 1.3B from the paper:
    https://arxiv.org/pdf/2307.08621v3.pdf
    """
    return TransformerLM(
        num_tokens=num_tokens,
        d_model=2048,
        nhead=8,
        num_layers=24,
        dim_feedforward=4096,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def benchmark_inference_throughput(
    retnet: RetNet, transformer: TransformerLM, seq_lengths: Sequence[int]
) -> Tuple[List[float], List[float]]:
    retnet_throughputs: List[float] = []
    transformer_throughputs: List[float] = []

    print("\nBenchmarking inference throughput...")
    for seq_length in seq_lengths:
        torch.cuda.empty_cache()
        print(f"seq_length: {seq_length}")
        x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, seq_length), device=DEVICE)

        # Benchmark *recurrent* RetNet formulation for inference
        retnet_result = benchmark(
            retnet.forward_recurrent, x[:, 0], seq_idx=0, prev_states=[]
        )
        retnet_throughput = BATCH_SIZE / retnet_result.mean
        print(f"RetNet throughput: {retnet_throughput:.3f} tokens/s")
        # Benchmark *parallel* transformer for inference (with memory cache)
        _ = transformer(x, start_pos=0)  # warmup memory cache
        transformer_result = benchmark(transformer, x[:, -1:], start_pos=seq_length - 1)
        transformer_throughput = BATCH_SIZE / transformer_result.mean
        print(f"Transformer throughput: {transformer_throughput:.3f} tokens/s")

        retnet_throughputs.append(retnet_throughput)
        transformer_throughputs.append(transformer_throughput)

    return retnet_throughputs, transformer_throughputs


@torch.no_grad()
def measure_inference_memory(
    retnet: RetNet, transformer: TransformerLM, seq_lengths: Sequence[int]
) -> Tuple[List[float], List[float]]:
    retnet_memories: List[float] = []
    transformer_memories: List[float] = []

    print("\nMeasuring inference memory...")
    for seq_length in seq_lengths:
        torch.cuda.empty_cache()
        print(f"seq_length: {seq_length}")
        x = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, seq_length), device=DEVICE)

        # Measure *recurrent* RetNet formulation for inference
        retnet_result = profile(
            retnet.forward_recurrent, x[:, 0], seq_idx=0, prev_states=[]
        )
        retnet_memory_gib = retnet_result.peak / 2**30
        print(f"RetNet memory: {retnet_memory_gib:.3f} GiB")
        # Measure *parallel* transformer for inference (with memory cache)
        _ = transformer(x, start_pos=0)  # warmup memory cache
        transformer_result = profile(transformer, x[:, -1:], start_pos=seq_length - 1)
        transformer_memory_gib = transformer_result.peak / 2**30
        print(f"Transformer memory: {transformer_memory_gib:.3f} GiB")

        retnet_memories.append(retnet_memory_gib)
        transformer_memories.append(transformer_memory_gib)

    return retnet_memories, transformer_memories


if __name__ == "__main__":
    retnet = retnet_1_3b(NUM_TOKENS, device=DEVICE, dtype=DTYPE).eval()
    transformer = transformer_1_3b(NUM_TOKENS, device=DEVICE, dtype=DTYPE).eval()

    if torch.cuda.is_available():
        retnet_footprints, transformer_footprints = measure_inference_memory(
            retnet, transformer, seq_lengths=SEQ_LENGTHS
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=SEQ_LENGTHS,
                y=retnet_footprints,
                name="RetNet",
                mode="lines+markers",
                line={"color": "blue"},
                marker={"color": "blue"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=SEQ_LENGTHS,
                y=transformer_footprints,
                name="Transformer",
                mode="lines+markers",
                line={"color": "red"},
                marker={"color": "red"},
            )
        )
        fig.update_layout(
            title="Inference Memory Footprint",
            xaxis_title="Sequence Length",
            yaxis_title="GPU Memory (GiB)",
            xaxis={"tickmode": "array", "tickvals": SEQ_LENGTHS},
            # place legend at center-left
            legend={"x": 0.1, "y": 0.5},
        )
        fig.write_image(os.path.join("doc", "inference-memory.png"))
    else:
        print("Skipping GPU memory profiling, because CUDA is not available.")

    retnet_throughputs, transformer_throughputs = benchmark_inference_throughput(
        retnet, transformer, seq_lengths=SEQ_LENGTHS
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=SEQ_LENGTHS,
            y=retnet_throughputs,
            name="RetNet",
            mode="lines+markers",
            line={"color": "blue"},
            marker={"color": "blue"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=SEQ_LENGTHS,
            y=transformer_throughputs,
            name="Transformer",
            mode="lines+markers",
            line={"color": "red"},
            marker={"color": "red"},
        )
    )
    fig.update_layout(
        title="Inference Throughput",
        xaxis_title="Sequence Length",
        yaxis_title="Throughput (tokens/s)",
        xaxis={"tickmode": "array", "tickvals": SEQ_LENGTHS},
        # place legend at center-left
        legend={"x": 0.1, "y": 0.5},
    )
    fig.write_image(os.path.join("doc", "inference-throughput.png"))

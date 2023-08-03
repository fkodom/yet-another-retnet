import asyncio
from typing import Callable, NamedTuple

import torch

KiB = 2**10
MiB = 2**20
GiB = 2**30


def _bytes_to_string(bytes: float) -> str:
    if bytes > GiB:
        return f"{bytes / GiB:.2f} GiB"
    elif bytes > MiB:
        return f"{bytes / MiB:.2f} MiB"
    elif bytes > KiB:
        return f"{bytes / KiB:.2f} KiB"
    else:
        return f"{bytes:.2f} Bytes"


class Profile(NamedTuple):
    name: str
    peak: float
    mean: float
    median: float

    def __str__(self, indent=""):
        return (
            f"{indent}{self.name}("
            f"peak: {_bytes_to_string(self.peak)}, "
            f"mean: {_bytes_to_string(self.mean)}, "
            f"median: {_bytes_to_string(self.median)})"
        )


async def _run_async(fn: Callable, *args, **kwargs):
    def synchronous_fn():
        torch.cuda.synchronize()
        _ = fn(*args, **kwargs)
        torch.cuda.synchronize()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, synchronous_fn)


async def _profile(fn: Callable, *args, interval: float = 1e-4, **kwargs):
    done = False
    memories = []

    async def _watch_cuda_memory(device=None):
        nonlocal done, memories
        while not done:
            memory = torch.cuda.memory_allocated()
            memories.append(memory)
            await asyncio.sleep(interval)

    cuda_task = asyncio.create_task(_watch_cuda_memory())
    main = asyncio.create_task(_run_async(fn, *args, **kwargs))
    done, _ = await main, cuda_task

    return memories


def profile(fn: Callable, *args, interval: float = 1e-4, **kwargs) -> Profile:
    torch.cuda.empty_cache()
    memories = torch.as_tensor(
        asyncio.run(_profile(fn, *args, interval=interval, **kwargs)),
        dtype=torch.float32,
    )
    return Profile(
        name="cuda",
        peak=torch.max(memories).item(),
        mean=torch.mean(memories).item(),
        median=torch.median(memories).item(),
    )

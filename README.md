# yet-another-retnet

(Work in progress) A simple but robust PyTorch implementation of RetNet from [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf).

<img src="doc/retnet-scaling.jpeg" alt="compare-attention-mechanisms" width="600"/>


### TODO

- [x] Equivalent **parallel** and **recursive** retention methods.  See: [retention.py](yet_another_retnet/retention.py)
- [x] `MultiheadRetention` module.  See: [retention.py](yet_another_retnet/retention.py)
- [ ] Equivalent **chunkwise** retention method.
- [ ] Recurrent position embedding implementation.
- [ ] End-to-end `RetNet` module.
- [ ] Reproduce inference memory, throughput, and latency benchmarks.
- [ ] Release stable version on PyPI.


## Install

```bash
pip install "yet-another-retnet @ git+ssh://git@github.com/fkodom/yet-another-retnet.git"
```

For contributors:
```bash
# Install all dev dependencies (tests etc.)
pip install "yet-another-retnet[test] @ git+ssh://git@github.com/fkodom/yet-another-retnet.git"
# Setup pre-commit hooks
pre-commit install
```

## Usage


### MultiheadRetention

Showing equivalent parallel and recurrent usage:

```python
import torch

from yet_another_retnet.retention import MultiheadRetention

mhr = MultiheadRetention(embed_dim=32, num_heads=4, device="cuda")

# input shape: (batch_size, seq_len, embed_dim)
q = k = v = torch.randn(1, 16, 32, device="cuda")

# Parallel retention
y_parallel, _ = mhr.forward_parallel(q, k, v)

# Recursive retention
outputs = []
prev_state = None
for i in range(32):
    out, prev_state = mhr.forward_recurrent(q[:, i], k[:, i], v[:, i], prev_state)
    outputs.append(out)
y_recursive = torch.stack(outputs, dim=1)

# Check that outputs are equal
torch.testing.assert_close(y_parallel, y_recursive)
```


### Retention forward pass

Similar to the example above, but head projections are not internalized by `MultiheadRetention`:

```python
import torch

from yet_another_retnet.retention import retention_parallel, retention_recurrent

# input shape: (batch_size, num_heads, seq_len, head_dim)
q = k = v = torch.randn(1, 4, 32, 8, device="cuda")

# Parallel retention
y_parallel, _ = retention_parallel(q, k, v)

# Recursive retention
outputs = []
prev_state = None
for i in range(32):
    out, prev_state = retention_recurrent(q[:, :, i], k[:, :, i], v[:, :, i], prev_state)
    outputs.append(out)
y_recursive = torch.stack(outputs, dim=2)

# Check that outputs are equal
torch.testing.assert_close(y_parallel, y_recursive)
```


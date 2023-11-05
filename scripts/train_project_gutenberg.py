import os
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import tiktoken
import torch
from lightning import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from yet_another_retnet.retnet import RetNet
from yet_another_retnet.utils.gutenberg import project_gutenberg_top_100_datapipe

torch.set_float32_matmul_precision("medium")
TOKENIZER = tiktoken.get_encoding("gpt2")
EVAL_PROMPT = "A Lannister always pays his debts."


def collate_fn(
    batch: List[str],
    max_length: int = 1024,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor]:
    x = torch.zeros(len(batch), max_length, device=device, dtype=torch.long)
    y = torch.zeros(len(batch), max_length, device=device, dtype=torch.long)

    for i, text in enumerate(batch):
        encoding = torch.as_tensor(
            TOKENIZER.encode(text), device=device, dtype=torch.long
        )
        seq_length = min(len(encoding) - 1, max_length)
        x[i, :seq_length] = encoding[:seq_length]
        y[i, :seq_length] = encoding[1 : seq_length + 1]

    return x, y


@dataclass
class TrainingState:
    fabric: Fabric
    model: RetNet
    optimizer: torch.optim.Optimizer
    callbacks: Sequence[Callable[["TrainingState", float], None]] = ()

    current_step: int = 0
    current_epoch: int = 0
    accumulate_grad_batches: int = 1
    monitor: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"


@dataclass
class ModelCheckpoint:
    state_dict: Dict[str, Tensor]
    optimizer_state: Dict[str, Tensor]
    current_step: int
    current_epoch: int

    @classmethod
    def from_training_state(cls, state: TrainingState) -> "ModelCheckpoint":
        return cls(
            state_dict=state.model.state_dict(),
            optimizer_state=state.optimizer.state_dict(),
            current_step=state.current_step,
            current_epoch=state.current_epoch,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_dict": self.state_dict,
            "optimizer_state": self.optimizer_state,
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
        }

    def save(self, path: str) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> "ModelCheckpoint":
        checkpoint_dict = torch.load(path)
        return cls(**checkpoint_dict)


class CheckpointCallback:
    def __init__(
        self, save_dir: str, name: str = "checkpoint_epoch-{epoch:03d}.pt"
    ) -> None:
        self.save_dir = save_dir
        self.name = name
        self.best_path: Optional[str] = None
        self.best_loss: Optional[float] = None

    def __call__(self, state: TrainingState, loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = loss

        fabric = state.fabric
        # 'local_rank == 0' means this only happens for the main process
        if fabric.local_rank == 0 and loss <= self.best_loss:
            checkpoint = ModelCheckpoint.from_training_state(state)
            self.best_loss = loss
            if self.best_path is not None:
                os.remove(self.best_path)
            self.best_path = os.path.join(
                self.save_dir, self.name.format(epoch=state.current_epoch)
            )
            torch.save(checkpoint, self.best_path)

        # All processes wait for main to finish saving the checkpoint.
        fabric.barrier()


def train_one_epoch(
    state: TrainingState,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    log_frequency: int = 25,
) -> None:
    state.current_epoch += 1
    fabric, model, optimizer = state.fabric, state.model, state.optimizer
    is_main_process = fabric.local_rank == 0
    is_training = model.training
    model.train()

    with tqdm(
        desc=f"Ep: {state.current_epoch}", disable=(not is_main_process)
    ) as progbar:
        train_loss, val_loss = 0.0, 0.0
        for x, y in train_dataloader:
            state.current_step += 1
            accumulating = state.current_step % state.accumulate_grad_batches != 0
            with fabric.no_backward_sync(model, enabled=accumulating):  # type: ignore
                loss = model.forward(inputs=x, labels=y)
                fabric.backward(loss)

            if not accumulating:
                optimizer.step()
                optimizer.zero_grad()

            if state.current_step % log_frequency == 0:
                fabric.log("loss", loss, step=state.current_step)
                train_loss = loss.item()
                progbar.set_postfix_str(f"loss={train_loss:.4f}", refresh=False)
            progbar.update(1)

        model.eval()
        val_progbar = tqdm(desc="val", position=1, leave=False)
        for i, (x, y) in enumerate(val_dataloader):
            with torch.inference_mode():
                loss = model.forward(inputs=x, labels=y)
            val_loss = (val_loss * i + loss.item()) / (i + 1)

            if i % log_frequency == 0:
                val_progbar.set_postfix_str(f"val_loss={val_loss:.4f}", refresh=False)
            val_progbar.update(1)
            progbar.update(1)

        fabric.log("val_loss", val_loss, step=state.current_step)
        val_progbar.close()
        progbar.set_postfix_str(
            f"loss={train_loss:.4f}, val_loss={val_loss:.4f}", refresh=False
        )

        for callback in state.callbacks:
            callback(state, val_loss)

        # Return model to its original training state
        model.train(mode=is_training)


def train(
    retnet: RetNet,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    accelerator: str = "auto",
    strategy: str = "auto",
    precision: Optional[str] = None,
    epochs: int = 10,
    lr: float = 3e-4,
    log_frequency: int = 25,
):
    if precision is None:
        if torch.cuda.is_available():
            # use bfloat16 if supported
            version, _ = torch.cuda.get_device_capability()
            precision = "bf16-mixed" if version >= 8 else "16-mixed"
        else:
            precision = "32-true"

    logger = TensorBoardLogger(root_dir="./")
    fabric = Fabric(
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,  # type: ignore
        loggers=[logger],
    )
    fabric.launch()
    print(f"Experiment version: {logger.version}")
    print("-" * 40)

    # Setup with fabric.
    optimizer = torch.optim.AdamW(retnet.parameters(), lr=lr)
    retnet, optimizer = fabric.setup(retnet, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    # Construct a training state and run the training loop.
    state = TrainingState(
        fabric=fabric,
        model=retnet,
        optimizer=optimizer,
        callbacks=[CheckpointCallback(save_dir=logger.log_dir)],
    )
    for _ in range(epochs):
        train_one_epoch(
            state=state,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            log_frequency=log_frequency,
        )


def generate(
    retnet: RetNet,
    prompt: str,
    prompt_chunk_size: Optional[int] = None,
    max_new_tokens: int = 4096,
    stop_tokens: Sequence[str] = (),
    top_k: int = 10,
    temperature: float = 1.0,
    seed: int = 42,
) -> Iterator[str]:
    seed_everything(seed)
    device = next(iter(retnet.parameters())).device
    is_training = retnet.training
    retnet.eval()

    # Tokenize the prompt and convert to a tensor.
    tokenized = TOKENIZER.encode(prompt)
    x = torch.as_tensor(tokenized, dtype=torch.long, device=device).unsqueeze_(0)

    if not prompt_chunk_size:
        prompt_chunk_size = x.size(1)

    prev_states: List[Optional[Tensor]] = [None] * retnet.num_layers
    start_idx: int = 0
    for start_idx in range(0, x.size(1), prompt_chunk_size):
        y, prev_states = retnet.forward_chunkwise(  # type: ignore
            x, start_idx=start_idx, prev_states=prev_states
        )
        y = y[:, -1]

    # Generate tokens until we reach the maximum number of tokens or a stop token.
    for i in range(max_new_tokens):
        probs: Tensor = torch.softmax(y.squeeze() / max(temperature, 1e-8), dim=-1)
        # Get top-k tokens, renormalize their probabilities, and weighted sample.
        tokens: Tensor  # for mypy
        probs, tokens = probs.topk(k=top_k, dim=-1)
        probs /= probs.sum()

        # Take weighted random sample from the top-k tokens.
        sampled_idx: int = torch.multinomial(probs, num_samples=1).item()  # type: ignore
        token: int = tokens[sampled_idx].item()  # type: ignore
        tokenized.append(token)
        yield TOKENIZER.decode(tokenized)

        token_str: str = TOKENIZER.decode([token])
        if token_str in stop_tokens:
            break
        elif i < (max_new_tokens - 1):
            start_idx += 1
            x = torch.as_tensor([token], dtype=torch.long, device=device)
            y, prev_states = retnet.forward_recurrent(  # type: ignore
                x, start_idx, prev_states=prev_states
            )

    # Restore the model's original training state.
    retnet.train(mode=is_training)


def main(
    model_checkpoint: Optional[str] = None,
    accelerator: str = "auto",
    strategy: str = "auto",
    precision: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    log_frequency: int = 25,
    seed: int = 42,
    eval_only: bool = False,
    eval_prompt: str = EVAL_PROMPT,
    eval_max_tokens: int = 1024,
):
    seed_everything(seed)
    # Create a (relatively small) model and dataloaders
    retnet = RetNet(
        num_tokens=TOKENIZER.n_vocab,
        d_model=768,
        nhead=8,
        num_layers=12,
    )
    if model_checkpoint is not None:
        retnet.load_state_dict(ModelCheckpoint.load(model_checkpoint).state_dict)

    if not eval_only:
        num_devices = torch.cuda.device_count()
        if num_devices > 0:
            # Lightning Fabric does not scale the batch size for distributed training.
            # In order to keep batch size the same, divide by the number of devices.
            if batch_size % num_devices != 0:
                raise ValueError(f"{batch_size=} must be divisible by {num_devices=}.")
            batch_size = batch_size // num_devices

        train_dataloader = DataLoader(
            project_gutenberg_top_100_datapipe(
                split="train",
                chunk_size=4096,
                step_size=1024,
                shuffle=True,
                drop_last=True,
            ),
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            project_gutenberg_top_100_datapipe(
                split="val", chunk_size=4096, step_size=1024
            ),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        train(
            retnet=retnet,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            accelerator=accelerator,
            strategy=strategy,
            precision=precision,
            epochs=epochs,
            lr=lr,
            log_frequency=log_frequency,
        )

    # Generate some text
    prev_output: str = ""
    for output in generate(retnet, eval_prompt, max_new_tokens=eval_max_tokens):
        # Return to the start of the line and print the output (no newline)
        print(output[len(prev_output) :], end="", flush=True)
        prev_output = output
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-frequency", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-prompt", type=str, default=EVAL_PROMPT)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)
    args = parser.parse_args()

    main(**vars(args))

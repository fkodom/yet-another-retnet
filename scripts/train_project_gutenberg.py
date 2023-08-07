import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

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

        if loss <= self.best_loss:
            checkpoint = ModelCheckpoint.from_training_state(state)
            self.best_loss = loss
            if self.best_path is not None:
                os.remove(self.best_path)
            self.best_path = os.path.join(
                self.save_dir, self.name.format(epoch=state.current_epoch)
            )
            torch.save(checkpoint, self.best_path)


def train_one_epoch(
    state: TrainingState,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    log_frequency: int = 25,
) -> None:
    fabric, model, optimizer = state.fabric, state.model, state.optimizer
    is_training = model.training
    model.train()

    with tqdm(desc=f"Ep: {state.current_epoch}") as progbar:
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
                val_progbar.set_postfix_str(f"{val_loss:.4f}", refresh=False)
            val_progbar.update(1)
            progbar.update(1)

        fabric.log("val_loss", val_loss, step=state.current_step)
        val_progbar.close()
        progbar.set_postfix_str(
            f"loss={train_loss:.4f}, val_loss={val_loss:.4f}", refresh=False
        )

        state.current_epoch += 1
        for callback in state.callbacks:
            callback(state, val_loss)

        # Return model to its original training state
        model.train(mode=is_training)


def main(
    accelerator: str = "auto",
    strategy: str = "auto",
    precision: Optional[str] = None,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 3e-4,
    log_frequency: int = 25,
):
    if precision is None:
        if torch.cuda.is_available():
            # use bfloat16 if supported
            version, _ = torch.cuda.get_device_capability()
            precision = "bf16-mixed" if version >= 8 else "16-mixed"
        else:
            precision = "float32"

    seed_everything(42)
    logger = TensorBoardLogger(root_dir="./")

    fabric = Fabric(
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,  # type: ignore
        loggers=[logger],
    )
    fabric.launch()
    print(f"Using accelerator: {fabric.accelerator}")
    print(f"Experiment version: {logger.version}")
    print("-" * 40)

    # Create a (fairly small) model and optimizer.
    retnet = RetNet(num_tokens=TOKENIZER.n_vocab, dim_feedforward=1024, num_layers=4)
    optimizer = torch.optim.AdamW(retnet.parameters(), lr=lr)
    # Create the dataloaders, and setup with fabric.
    train_dataloader = DataLoader(
        project_gutenberg_top_100_datapipe(
            split="train", chunk_size=4096, step_size=1024, shuffle=True, drop_last=True
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
    # Setup with fabric.
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


if __name__ == "__main__":
    main()

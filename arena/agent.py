from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arena.genome import Genome

ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}

OPTIMIZER_MAP: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class AgentStats:
    """Mutable performance statistics accumulated across rounds."""

    wins: int = 0
    losses: int = 0
    rounds_survived: int = 0
    best_accuracy: float = 0.0
    generation: int = 0


class Agent:
    """A single competitor in the arena: a PyTorch model driven by a Genome."""

    def __init__(
        self,
        genome: Genome,
        input_size: int,
        num_classes: int,
        *,
        agent_id: str | None = None,
        name: str | None = None,
        generation: int = 0,
        parents: list[str] | None = None,
        lineage: list[dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.id = agent_id or _make_id()
        self.name = name or f"Tribute_{self.id[:4]}"
        self.genome = genome
        self.input_size = input_size
        self.num_classes = num_classes
        self.parents = parents or []
        self.lineage = lineage or []
        self.stats = AgentStats(generation=generation)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = self._build_model()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Sequential:
        """Construct an MLP from the genome specification."""
        activation_cls = ACTIVATION_MAP.get(self.genome.activation, nn.ReLU)
        layers: list[nn.Module] = []

        prev_size = self.input_size
        for hidden_size in self.genome.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_cls())
            if self.genome.dropout > 0:
                layers.append(nn.Dropout(self.genome.dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.num_classes))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataloader: DataLoader, epochs: int | None = None) -> list[float]:
        """Train the model and return per-epoch average loss."""
        epochs = epochs or self.genome.epochs_per_round

        optimizer_cls = OPTIMIZER_MAP.get(self.genome.optimizer, torch.optim.Adam)
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.genome.learning_rate,
            weight_decay=self.genome.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_losses: list[float] = []

        for _ in range(epochs):
            running_loss = 0.0
            n_batches = 0
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if inputs.dim() > 2:
                    inputs = inputs.view(inputs.size(0), -1)

                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

        return epoch_losses

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the model and return accuracy and loss."""
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)

            logits = self.model(inputs)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        self.stats.best_accuracy = max(self.stats.best_accuracy, accuracy)

        return {"accuracy": accuracy, "loss": avg_loss}

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def clone(self, *, mutate: bool = True) -> Agent:
        """Create a child agent, optionally mutating the genome."""
        child_genome = self.genome.mutate() if mutate else Genome.from_dict(self.genome.to_dict())

        child = Agent(
            genome=child_genome,
            input_size=self.input_size,
            num_classes=self.num_classes,
            generation=self.stats.generation + 1,
            parents=[self.id],
            lineage=self.lineage + [{"parent": self.id, "op": "clone", "mutated": mutate}],
            device=self.device,
        )
        child.name = f"Heir_of_{self.id[:4]}"
        return child

    def crossbreed(self, other: Agent) -> Agent:
        """Create a child agent by crossing over genomes from two parents."""
        child_genome = self.genome.crossover(other.genome)

        child = Agent(
            genome=child_genome,
            input_size=self.input_size,
            num_classes=self.num_classes,
            generation=max(self.stats.generation, other.stats.generation) + 1,
            parents=[self.id, other.id],
            lineage=self.lineage + [
                {"parents": [self.id, other.id], "op": "crossover"},
            ],
            device=self.device,
        )
        child.name = f"Cross_{self.id[:4]}x{other.id[:4]}"
        return child

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        acc = f"{self.stats.best_accuracy:.2%}"
        arch = "x".join(str(s) for s in self.genome.hidden_layers)
        return f"Agent({self.name}, arch=[{arch}], best_acc={acc})"

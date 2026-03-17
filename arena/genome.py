from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from config import ArenaConfig

OPTIMIZERS = ["adam", "sgd", "rmsprop"]
ACTIVATIONS = ["relu", "tanh", "gelu", "leaky_relu"]
BATCH_SIZES = [16, 32, 64, 128, 256]

LR_RANGE = (1e-5, 1e-1)
DROPOUT_RANGE = (0.0, 0.5)
WEIGHT_DECAY_RANGE = (0.0, 1e-2)
HIDDEN_SIZE_RANGE = (16, 512)
HIDDEN_DEPTH_RANGE = (1, 5)
EPOCHS_RANGE = (1, 20)

_DEFAULT_MUTATION_RATE = ArenaConfig().mutation_rate


def _log_uniform(lo: float, hi: float) -> float:
    """Sample from a log-uniform distribution over [lo, hi]."""
    return math.exp(random.uniform(math.log(lo), math.log(hi)))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(slots=True)
class Genome:
    """Inheritable configuration encoding architecture and hyperparameters."""

    learning_rate: float = 1e-3
    optimizer: str = "adam"
    batch_size: int = 64
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    weight_decay: float = 1e-4
    epochs_per_round: int = 5

    # --- factories -------------------------------------------------------

    @classmethod
    def random(cls) -> Genome:
        """Create a genome with fully randomised parameters."""
        depth = random.randint(*HIDDEN_DEPTH_RANGE)
        layers = [2 ** random.randint(4, 9) for _ in range(depth)]
        return cls(
            learning_rate=_log_uniform(*LR_RANGE),
            optimizer=random.choice(OPTIMIZERS),
            batch_size=random.choice(BATCH_SIZES),
            hidden_layers=layers,
            activation=random.choice(ACTIVATIONS),
            dropout=random.uniform(*DROPOUT_RANGE),
            weight_decay=random.uniform(*WEIGHT_DECAY_RANGE),
            epochs_per_round=random.randint(*EPOCHS_RANGE),
        )

    # --- genetic operators -----------------------------------------------

    def mutate(self, mutation_rate: float = _DEFAULT_MUTATION_RATE) -> Genome:
        """Return a new Genome by applying Gaussian noise to continuous
        parameters and random resampling to categorical parameters."""

        lr = self.learning_rate
        if random.random() < mutation_rate:
            log_lr = math.log(lr) + random.gauss(0, 0.5)
            lr = _clamp(math.exp(log_lr), *LR_RANGE)

        dropout = self.dropout
        if random.random() < mutation_rate:
            dropout = _clamp(dropout + random.gauss(0, 0.05), *DROPOUT_RANGE)

        wd = self.weight_decay
        if random.random() < mutation_rate:
            wd = _clamp(wd + random.gauss(0, 0.001), *WEIGHT_DECAY_RANGE)

        epochs = self.epochs_per_round
        if random.random() < mutation_rate:
            epochs = int(_clamp(epochs + random.choice([-1, 0, 1]), *EPOCHS_RANGE))

        optimizer = self.optimizer
        if random.random() < mutation_rate:
            optimizer = random.choice(OPTIMIZERS)

        activation = self.activation
        if random.random() < mutation_rate:
            activation = random.choice(ACTIVATIONS)

        bs = self.batch_size
        if random.random() < mutation_rate:
            bs = random.choice(BATCH_SIZES)

        layers = list(self.hidden_layers)
        if random.random() < mutation_rate and layers:
            idx = random.randrange(len(layers))
            layers[idx] = int(_clamp(
                layers[idx] + random.choice([-32, -16, 16, 32]),
                *HIDDEN_SIZE_RANGE,
            ))
        if random.random() < mutation_rate * 0.5:
            if len(layers) > HIDDEN_DEPTH_RANGE[0] and random.random() < 0.5:
                layers.pop(random.randrange(len(layers)))
            elif len(layers) < HIDDEN_DEPTH_RANGE[1]:
                layers.insert(
                    random.randrange(len(layers) + 1),
                    2 ** random.randint(4, 9),
                )

        return Genome(
            learning_rate=lr,
            optimizer=optimizer,
            batch_size=bs,
            hidden_layers=layers,
            activation=activation,
            dropout=dropout,
            weight_decay=wd,
            epochs_per_round=epochs,
        )

    def crossover(self, other: Genome) -> Genome:
        """Produce a child genome by uniformly mixing genes from two parents."""

        def pick(a, b):  # noqa: E731
            return a if random.random() < 0.5 else b

        if random.random() < 0.5:
            layers = list(self.hidden_layers)
        else:
            layers = list(other.hidden_layers)

        return Genome(
            learning_rate=pick(self.learning_rate, other.learning_rate),
            optimizer=pick(self.optimizer, other.optimizer),
            batch_size=pick(self.batch_size, other.batch_size),
            hidden_layers=layers,
            activation=pick(self.activation, other.activation),
            dropout=pick(self.dropout, other.dropout),
            weight_decay=pick(self.weight_decay, other.weight_decay),
            epochs_per_round=pick(self.epochs_per_round, other.epochs_per_round),
        )

    # --- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize this genome to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        """Deserialize a genome from a dictionary."""
        return cls(**data)

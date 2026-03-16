from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GameType = type[Any] | str


@dataclass(slots=True)
class ArenaConfig:
    """Primary runtime configuration for the Darwin Arena."""

    population_size: int = 20
    n_generations: int = 10
    task: str = "mnist"
    rounds: list[GameType] = field(
        default_factory=lambda: [
            "AccuracyTrial",
            "HeadToHead",
            "AdversarialRound",
            "Gauntlet",
        ]
    )
    elimination_rates: list[float] = field(
        default_factory=lambda: [0.4, 0.5, 0.3, 0.0]
    )
    mutation_rate: float = 0.2
    crossover_rate: float = 0.3
    resurrection_rate: float = 0.0
    seed: int | None = None
    output_dir: str = "output"

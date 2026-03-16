Darwin Arena is a multi-agent competitive training system inspired by tournament-style
survival dynamics (e.g., Hunger Games / Squid Game). Instead of a human experimenting
with one model at a time, a population of agents competes for survival, and the
resulting selection pressure drives emergent optimization of model architectures and
hyperparameters.

## Core Concept

Each agent is a small neural network paired with a genome: a configuration that defines
its architecture and hyperparameters. Agents participate in structured elimination
rounds. Agents with poor performance are removed from the population, while successful
agents reproduce (with mutation and optional crossover). Over many generations, fitter
agents emerge without explicit human-guided hyperparameter search.

## Project Structure

The reference implementation assumes the following project structure:

darwin_arena/
├── arena/
│   ├── arena.py          # Main Arena class, manages seasons/rounds
│   ├── agent.py          # Agent class: model + genome + lineage tracking
│   ├── genome.py         # Genome dataclass, mutation, crossover logic
│   └── selection.py      # SelectionEngine: elimination, reproduction, harvesting
├── games/
│   ├── base_game.py      # Abstract Game class
│   ├── accuracy_trial.py # Round 1: raw task accuracy, bottom % eliminated
│   ├── head_to_head.py   # Round 2: paired agents, same data, winner advances
│   ├── adversarial.py    # Round 3: one agent generates hard examples, one solves
│   └── gauntlet.py       # Final: distribution-shifted held-out evaluation
├── tasks/
│   ├── base_task.py      # Abstract task
│   ├── mnist_task.py     # MNIST classification (fast baseline)
│   └── synthetic_task.py # Configurable synthetic regression/classification
├── visualization/
│   ├── bracket.py        # Tournament bracket renderer (rich/ascii art)
│   ├── lineage.py        # Evolutionary tree (show parent/child mutations)
│   └── fitness_plot.py   # Live fitness curves per generation
├── config.py             # ArenaConfig dataclass
├── run_arena.py          # CLI entrypoint
└── README.md

## Genome

The genome is the inheritable configuration that is mutated across generations. It
encodes at least the following fields:

- learning_rate: float (typically sampled log-uniformly in the range 1e-5 to 1e-1)
- optimizer: str (e.g., `adam`, `sgd`, `rmsprop`)
- batch_size: int (e.g., 16, 32, 64, 128, 256)
- hidden_layers: List[int] (e.g., `[128, 64]` for a two-layer MLP)
- activation: str (e.g., `relu`, `tanh`, `gelu`, `leaky_relu`)
- dropout: float (0.0 to 0.5)
- weight_decay: float (0.0 to 1e-2)
- epochs_per_round: int (number of training epochs executed per round)

Typical genome operations include:

- `mutate(mutation_rate)`: produces a new genome by applying Gaussian noise to
  continuous parameters and random flips or resampling to categorical parameters.
- `crossover(other_genome)`: produces a child genome by mixing genes from two parents.
- `to_dict()` / `from_dict()`: serializes and deserializes genome instances.

## Agent

An agent represents a single model instance participating in the arena. Each agent
encapsulates:

- A PyTorch model whose architecture is derived from its genome.
- Its genome instance.
- A lineage structure (e.g., list of parent agent IDs and mutation descriptors).
- Statistics such as wins, losses, rounds survived, best accuracy, and generation index.
- A unique identifier and an optional human-readable name (e.g., `Tribute_7`,
  `Heir_of_23`).

Representative agent methods include:

- `build_model()`: constructs a network from the genome specification.
- `train(dataloader, epochs)`: trains the model for a given number of epochs and
  returns a training loss curve or equivalent training metrics.
- `evaluate(dataloader)`: evaluates the model and returns metrics such as accuracy and
  loss.
- `clone(mutate=True)`: creates a new agent sharing the same lineage, with an optional
  genome mutation step.
- `crossbreed(other_agent)`: creates a new agent via crossover of genomes from two
  parent agents.

## Games

### AccuracyTrial (Red Light Green Light)

The AccuracyTrial round evaluates agents purely on raw task accuracy:

- All agents train on the same training split.
- Agents are evaluated on a validation split.
- The bottom \(X\%\) of agents (configurable, default 40%) are eliminated.
- The top-performing agents receive a "survival bonus", typically implemented as
  additional training epochs in the next round.

### HeadToHead (Tug of War)

The HeadToHead round compares agents in pairwise matches:

- Agents are randomly paired into matches.
- Each pair trains on identical data.
- The agent in each pair with higher validation accuracy advances; the other is
  eliminated.
- Ties are broken based on training efficiency (e.g., fewer epochs or lower
  computational cost).

### AdversarialRound (Glass Bridge)

The AdversarialRound introduces adversarial pressure between agents:

- Surviving agents are partitioned into generators and solvers.
- Generator agents are trained to produce examples that are difficult for the solvers
  (e.g., induce high loss).
- Solver agents are evaluated on their ability to correctly classify or solve these
  hard examples.
- Solvers that outperform the associated generators survive, and generators that
  successfully fool solvers also survive, creating mutual selection pressure.

### Gauntlet (Final)

The Gauntlet round provides a final robustness-oriented evaluation:

- Agents are evaluated on held-out data exhibiting distribution shifts (e.g., noise,
  rotations, label perturbations).
- All remaining agents are scored and ranked by a robustness metric.
- A final leaderboard is produced summarizing the performance of surviving agents.

## SelectionEngine

The `SelectionEngine` manages the population lifecycle between generations.

Its responsibilities include:

- `eliminate(agents, bottom_pct)`: removes the worst-performing portion of the
  population.
- `reproduce(survivors, target_population_size)`: replenishes the population via:
  - cloning top agents with mutation,
  - crossbreeding pairs of survivors, and
  - optionally introducing "resurrection" agents (random new genomes to maintain
    genetic diversity).
- `harvest(dead_agents)`: extracts high-performing gene segments from eliminated
  agents and stores them in a gene pool that can be sampled during reproduction.
- `log_generation(generation_stats)`: records snapshots of each generation for
  downstream visualization and analysis.

## Arena

The `Arena` class orchestrates the execution of seasons and generations. A typical
configuration-based instantiation is illustrated below:
```python
arena = Arena(config=ArenaConfig(
    population_size=20,
    n_generations=10,
    task="mnist",
    rounds=[AccuracyTrial, HeadToHead, AdversarialRound, Gauntlet],
    elimination_rates=[0.4, 0.5, 0.3, 0.0],  # per round
))

arena.run()
```

In this setup, a season corresponds to one full sequence of rounds followed by
reproduction. The `run()` method iterates over generations, executing the configured
round sequence for each generation. After the final generation, the system can produce
a "Hall of Fame" view summarizing the top agents and their full lineages.

## Visualization

Visualization is designed to make each round of the arena observable and engaging,
primarily via terminal output (e.g., using `rich`) and summary plots.

Typical visualization features include:

- Round announcement banners describing the current phase of the tournament.
- Live progress indicators per agent during training.
- Post-round leaderboards showing rank, agent name, accuracy, generation, and parent
  lineage.
- Elimination summaries indicating which agents were removed and relevant metrics
  (e.g., `"Tribute_12 eliminated — accuracy 0.43, bottom 40%"`).
- Reproduction summaries listing newly created agents and their parents.
- A final evolutionary lineage tree rendered as ASCII art, visualizing parent–child
  chains across generations.
- Fitness curves plotted across generations (e.g., via matplotlib), typically saved to
  files such as `output/fitness_{timestamp}.png`.

## Config

The `ArenaConfig` object defines the primary runtime configuration of the system. It
includes at least the following fields:
- `population_size` (default 20)
- `n_generations` (default 10)
- `task` (e.g., `mnist` or `synthetic`)
- `rounds` (list of game types to run each generation)
- `elimination_rates` (per-round elimination fractions)
- `mutation_rate` (default 0.2)
- `crossover_rate` (default 0.3)
- `resurrection_rate` (fraction of the new population consisting of randomly
  initialized agents)
- `seed` (for reproducibility)
- `output_dir` (for saving checkpoints and plots)

## CLI

The reference CLI entrypoint is `run_arena.py`. A representative invocation is:

python run_arena.py \
  --population 20 \
  --generations 10 \
  --task mnist \
  --mutation-rate 0.2 \
  --output ./arena_output \
  --seed 42

The CLI supports additional flags, such as:

- `--fast`, which reduces the number of epochs per round for quicker test runs.
- `--resume`, which loads a saved checkpoint and continues an existing run.

## Requirements

The system has the following primary dependencies and runtime characteristics:

- PyTorch, used for defining and training all models.
- `rich`, used for terminal visualization.
- `matplotlib`, used for generating fitness plots.
- `pydantic` or the Python `dataclasses` module, used for configuration and genome
  definitions.
- Python 3.10 or later.
- Support for execution on CPU within reasonable time for MNIST-scale tasks and small
  models.
- Optional GPU acceleration, automatically used when available.
# RL-Pruning: Reinforcement Learning for Transformer Compression

RL-Pruning is a framework for layer-wise, structured pruning of Transformers
using Reinforcement Learning.

An RL agent learns per-layer pruning decisions for FFN neurons and attention
heads while balancing accuracy vs. compression.

The current implementation targets BERT on GLUE/SST-2, but the architecture is model-agnostic.

## Structure

```
rl-pruning/
├── rl_pruning/
│   ├── assets/               # Baselines
│   ├── cli.py                # Typer CLI entrypoint
│   ├── config.py             # Hyperparameter configs
│   ├── envs/                 # Gym environments
│   ├── eval/                 # GLUE evaluation
│   ├── pruning/              # Uniform baseline pruning
│   ├── scripts/              # Pre-compute baseline
│   ├── models/               # PPO actor-critic
│   ├── training/             # Training + checkpointing
│   ├── analysis/             # Policy inspection + plots
│   └── utils/                # Model loading, helpers, priors
├── pyproject.toml
└── README.md
```

## Usage

### Train a pruning policy

```
uv run rl-pruning train --episodes 300 --alpha 0.1
```

#### Arguments

`--episodes`: PPO training episodes

`--alpha`: compression penalty coefficient

#### Validate the environment

```
uv run rl-pruning validate --episodes 5
```

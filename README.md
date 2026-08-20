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
│   ├── cli.py                # Typer CLI entrypoint
│   ├── config.py             # Hyperparameter + pruning configs
│   ├── envs/                 # Gym environment (RL + pruning logic)
│   ├── eval/                 # GLUE evaluation (+ partial/subset eval)
│   ├── models/               # PPO actor-critic
│   ├── training/             # PPO training + checkpointing
│   ├── analysis/             # Policy inspection + plots
│   └── utils/                # Model loading, priors
├── tests/                    # pytest suite
├── pyproject.toml
└── README.md
```

## Reward design

- **Terminal reward** (end of episode): `retention - alpha * compression`,
  where `retention = accuracy / original_accuracy` and
  `compression = 1 - remaining_params / original_params`. A soft floor at
  `acc_floor_ratio` (default `0.85`) adds a strong penalty if retention drops
  below it, discouraging catastrophic degradation.
- **Proxy/shaping reward** (per layer, no eval): rewards the per-step
  compression signal (`shaping_coef * param_reduction`) minus an
  importance-weighted pruning penalty (`shaping_penalty * importance * (...)`),
  aligned to the terminal reward's compression scale.

## Usage

### Train a pruning policy

```
uv run rl-pruning train --episodes 300 --alpha 0.3
```

#### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | `200` | PPO training episodes |
| `--alpha` | `0.3` | Compression penalty coefficient (terminal reward) |
| `--max-prune-ratio` | `0.8` | Upper bound on per-layer prune ratios |
| `--acc-floor-ratio` | `0.85` | Retention soft floor (terminal reward) |
| `--shaping-coef` | `1.0` | Proxy reward compression coefficient |
| `--shaping-penalty` | `0.1` | Proxy reward importance penalty coefficient |
| `--layer-weight-decay` | `0.0` | Optional monotonic layer weighting in GAE (0 disables) |
| `--n-rollouts` | `8` | Rollouts collected per PPO update |
| `--subset-frac` | `0.1` | Fraction of the eval set used per episode (partial eval) |
| `--eval-every` | `10` | Run full evaluation every N episodes |

### Validate the environment

```
uv run rl-pruning validate --episodes 5
```

## Tests

```
uv run pytest tests/
```

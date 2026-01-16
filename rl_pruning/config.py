from dataclasses import dataclass


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4


@dataclass
class PruningConfig:
    max_prune_ratio: float = 0.8
    alpha: float = 0.05

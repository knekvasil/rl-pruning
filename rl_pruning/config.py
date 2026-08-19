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
    n_rollouts: int = 8
    # Optional monotonic layer weighting in the GAE (0 disables).
    layer_weight_decay: float = 0.0


@dataclass
class PruningConfig:
    max_prune_ratio: float = 0.8
    alpha: float = 0.3
    acc_floor_ratio: float = 0.85
    # Proxy/shaping reward coefficients. shaping_coef scales the per-step
    # compression signal (aligned to the terminal reward's compression term);
    # shaping_penalty scales the importance/ratio penalty.
    shaping_coef: float = 1.0
    shaping_penalty: float = 0.1

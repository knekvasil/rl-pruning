import numpy as np
import torch
import pytest

from rl_pruning.training.trainer import train_ppo
from rl_pruning.config import PPOConfig
from rl_pruning.models.ppo import PPOActorCritic


@pytest.fixture(autouse=True)
def no_checkpoint_writes(monkeypatch):
    import rl_pruning.training.checkpointing as ckpt

    monkeypatch.setattr(ckpt.CheckpointManager, "save_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(ckpt.CheckpointManager, "should_stop", lambda self, r: False)


def test_multi_rollout_batch_size(stub_env, monkeypatch):
    """PPO update must run on n_rollouts * n_layers transitions."""
    seen = []
    orig_evaluate = PPOActorCritic.evaluate

    def spy(self, obs, action):
        seen.append(obs.shape[0])
        return orig_evaluate(self, obs, action)

    monkeypatch.setattr(PPOActorCritic, "evaluate", spy)

    cfg = PPOConfig(n_rollouts=3, epochs=1, lr=1e-3)
    train_ppo(env=stub_env, config=cfg, episodes=2, device="cpu")

    assert seen, "evaluate was never called"
    expected = 3 * stub_env.n_layers  # 3 rollouts * 12 layers
    assert all(b == expected for b in seen)


def test_reward_is_mean_over_rollouts(stub_env):
    cfg = PPOConfig(n_rollouts=3, epochs=1, lr=1e-3)
    _, history = train_ppo(env=stub_env, config=cfg, episodes=3, device="cpu")
    assert len(history["rewards"]) == 3
    expected = 0.5 * (stub_env.n_layers - 1) + 1.0  # sum per identical rollout
    assert all(abs(r - expected) < 1e-6 for r in history["rewards"])


def test_single_rollout_backward_compatible(stub_env):
    """n_rollouts=1 reproduces the original single-trajectory behaviour."""
    cfg = PPOConfig(n_rollouts=1, epochs=1, lr=1e-3)
    _, history = train_ppo(env=stub_env, config=cfg, episodes=2, device="cpu")
    assert len(history["rewards"]) == 2


def test_history_contains_expected_keys(stub_env):
    cfg = PPOConfig(n_rollouts=2, epochs=1, lr=1e-3)
    _, history = train_ppo(env=stub_env, config=cfg, episodes=1, device="cpu")
    for key in ("rewards", "accuracies", "compressions", "full_accuracies", "alpha"):
        assert key in history


def test_layer_weight_decay_disabled_by_default():
    """layer_weight_decay defaults to 0 (no layer weighting in GAE)."""
    cfg = PPOConfig()
    assert cfg.layer_weight_decay == 0.0


def test_trainer_runs_with_layer_weight_decay(stub_env):
    """Trainer accepts a nonzero layer_weight_decay without error."""
    cfg = PPOConfig(n_rollouts=2, epochs=1, lr=1e-3, layer_weight_decay=0.5)
    _, history = train_ppo(env=stub_env, config=cfg, episodes=2, device="cpu")
    assert len(history["rewards"]) == 2

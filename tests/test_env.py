import numpy as np
import torch
import pytest

from rl_pruning.envs.transformer_pruning_env import TransformerPruningEnv
import rl_pruning.envs.transformer_pruning_env as envmod


@pytest.fixture
def make_env(monkeypatch):
    def _make(model, eval_fn=None, **kwargs):
        if eval_fn is None:
            eval_fn = lambda m: 0.9
        return TransformerPruningEnv(
            model=model,
            tokenizer=None,
            eval_fn=eval_fn,
            device="cpu",
            **kwargs,
        )

    return _make


def run_episode(env, action=np.array([0.3, 0.2], dtype=np.float32)):
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(action)
    return env


def test_state_shape_non_done(make_env, stub_model):
    env = make_env(stub_model)
    obs, _ = env.reset()
    assert obs.shape == (5,)
    assert obs[0] < 1.0  # not terminal yet


def test_state_shape_terminal(make_env, stub_model):
    env = make_env(stub_model)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.array([0.2, 0.1], dtype=np.float32))
    obs = env._get_state()
    assert obs.shape == (5,)
    assert obs[0] == 1.0  # terminal marker
    # param ratio present and in [0,1]
    assert 0.0 <= obs[3] <= 1.0


def test_observation_space_matches_state(make_env, stub_model):
    env = make_env(stub_model)
    assert env.observation_space.shape == (5,)


def test_episode_prunes_all_layers(make_env, stub_model):
    env = make_env(stub_model)
    env.reset()
    n_steps = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(np.array([0.3, 0.2], dtype=np.float32))
        n_steps += 1
    assert n_steps == env.n_layers


def test_prune_ffn_reduces_dimensions(stub_model):
    env = TransformerPruningEnv(
        model=stub_model, tokenizer=None, eval_fn=lambda m: 0.9, device="cpu"
    )
    layer = env.model.bert.encoder.layer[0]
    before = layer.intermediate.dense.out_features
    env._prune_ffn(layer, 0.5)
    after = layer.intermediate.dense.out_features
    assert after == before // 2
    assert layer.output.dense.in_features == after


def test_prune_ffn_keeps_highest_importance_neurons(stub_model):
    env = TransformerPruningEnv(
        model=stub_model, tokenizer=None, eval_fn=lambda m: 0.9, device="cpu"
    )
    layer = env.model.bert.encoder.layer[0]
    fc1 = layer.intermediate.dense
    with torch.no_grad():
        fc1.weight.zero_()
        fc1.weight[0].fill_(1.0)  # neuron 0 has largest L1 norm (sum=in_features)
        fc1.weight[1].fill_(0.1)
    env._prune_ffn(layer, 0.5)  # keep half
    new_fc1 = layer.intermediate.dense
    # kept rows should include the high-norm neuron 0
    assert new_fc1.out_features == fc1.out_features // 2
    assert torch.allclose(new_fc1.weight[0].abs().sum(), torch.tensor(8.0))


def test_prune_heads_reduces_head_count(stub_model):
    env = TransformerPruningEnv(
        model=stub_model, tokenizer=None, eval_fn=lambda m: 0.9, device="cpu"
    )
    layer = env.model.bert.encoder.layer[0]
    before = layer.attention.self.num_attention_heads
    env._prune_heads(layer, 0.5)
    after = layer.attention.self.num_attention_heads
    assert after == before // 2


# -------------------- Track2: terminal reward --------------------


def _terminal_reward(env, accuracy):
    """Compute the terminal reward as the env would, with a given accuracy."""
    total_param_ratio = env._param_count / env.original_params
    compression = 1.0 - total_param_ratio
    retention = accuracy / env.original_accuracy
    reward = retention - env.alpha * compression
    if retention < env.acc_floor_ratio:
        reward -= 2.0
    return reward


def test_terminal_reward_formula(make_env, stub_model):
    """reward = retention - alpha * compression (no baseline, no dynamic penalty)."""
    env = make_env(stub_model, eval_fn=lambda m: 0.9)
    env.original_accuracy = 0.9
    env._param_count = int(env.original_params * 0.6)  # ~40% compression
    accuracy = 0.81  # 90% retention
    compression = 1.0 - env._param_count / env.original_params
    retention = accuracy / env.original_accuracy
    expected = retention - env.alpha * compression
    assert _terminal_reward(env, accuracy) == pytest.approx(expected)


def test_alpha_scales_compression_penalty(make_env, stub_model):
    """Higher alpha -> more negative compression penalty at terminal."""
    # disable the soft floor so only alpha differs
    env_hi = make_env(stub_model, alpha=0.5, acc_floor_ratio=0.0, eval_fn=lambda m: 0.9)
    env_lo = make_env(stub_model, alpha=0.05, acc_floor_ratio=0.0, eval_fn=lambda m: 0.9)
    for env in (env_hi, env_lo):
        env.original_accuracy = 0.9
        env._param_count = int(env.original_params * 0.6)  # ~40% compression
    compression = 1.0 - env_lo._param_count / env_lo.original_params
    acc = 0.7
    reward_lo = _terminal_reward(env_lo, acc)
    reward_hi = _terminal_reward(env_hi, acc)
    assert reward_hi < reward_lo
    # exact form: penalty == alpha * compression
    assert reward_lo == pytest.approx(acc / 0.9 - 0.05 * compression)
    assert reward_hi == pytest.approx(acc / 0.9 - 0.5 * compression)


def test_soft_floor_penalizes_catastrophic_retention(make_env, stub_model):
    """Retention below acc_floor_ratio triggers an extra -2.0 penalty."""
    # same accuracy, only the floor boundary differs
    env_no_floor = make_env(stub_model, acc_floor_ratio=0.0, eval_fn=lambda m: 0.9)
    env_floor = make_env(stub_model, acc_floor_ratio=0.9, eval_fn=lambda m: 0.9)
    for env in (env_no_floor, env_floor):
        env.original_accuracy = 0.9
        env._param_count = env.original_params  # zero compression
    acc = 0.8  # retention 0.889
    base = _terminal_reward(env_no_floor, acc)  # above floor -> no penalty
    penalized = _terminal_reward(env_floor, acc)  # 0.889 < 0.9 -> penalty
    assert penalized == pytest.approx(base - 2.0)


def test_default_alpha_applied_in_terminal_reward(make_env, stub_model):
    """alpha defaults to 0.3 and is used in terminal reward."""
    env = make_env(stub_model)
    assert env.alpha == 0.3


def test_acc_floor_ratio_configurable(make_env, stub_model):
    env = make_env(stub_model, acc_floor_ratio=0.9)
    assert env.acc_floor_ratio == 0.9


# -------------------- Track2: proxy (shaping) reward --------------------


def test_proxy_reward_uses_shaping_coef_and_penalty(make_env, stub_model):
    """Non-terminal shaping reward = coef*param_reduction - penalty*importance*..."""
    env = make_env(stub_model, shaping_coef=3.0, shaping_penalty=0.5)
    env.reset()

    action = np.array([0.3, 0.2], dtype=np.float32)
    pruned_layer_idx = env.current_layer
    params_before = env._param_count
    _, reward, done, _, _ = env.step(action)
    assert not done  # n_layers=3, only pruned layer 0

    removed = params_before - env._param_count
    param_reduction = removed / env.original_params
    importance = env.layer_importance[pruned_layer_idx]
    ffn_ratio, head_ratio = action
    expected = 3.0 * param_reduction - 0.5 * importance * (
        ffn_ratio**2 + head_ratio**2
    )
    assert reward == pytest.approx(expected)


def test_shaping_coef_defaults(make_env, stub_model):
    env = make_env(stub_model)
    assert env.shaping_coef == 1.0
    assert env.shaping_penalty == 0.1


def test_env_defaults_match_pruning_config(make_env, stub_model):
    """Env defaults must mirror PruningConfig (single source of truth)."""
    from rl_pruning.config import PruningConfig

    cfg = PruningConfig()
    env = make_env(stub_model)
    assert env.max_prune_ratio == cfg.max_prune_ratio
    assert env.alpha == cfg.alpha
    assert env.acc_floor_ratio == cfg.acc_floor_ratio
    assert env.shaping_coef == cfg.shaping_coef
    assert env.shaping_penalty == cfg.shaping_penalty


# -------------------- Track1C: incremental param counting --------------------


def test_incremental_param_count_matches_full_rescan(make_env, stub_model):
    """_param_count (incremental) must equal a full sum(p.numel()) re-scan."""
    env = make_env(stub_model)
    env.reset()
    assert env._param_count == env.original_params
    assert env._param_count == sum(p.numel() for p in env.model.parameters())

    done = False
    while not done:
        _, _, done, _, _ = env.step(np.array([0.3, 0.2], dtype=np.float32))
        assert env._param_count == sum(
            p.numel() for p in env.model.parameters()
        )


def test_prune_layer_updates_running_count(make_env, stub_model):
    """Pruning a layer removes exactly the reported params from the cache."""
    env = make_env(stub_model)
    env.reset()
    layer = env.model.bert.encoder.layer[0]
    before = env._param_count
    removed = env._prune_layer(layer, np.array([0.5, 0.5], dtype=np.float32))
    assert env._param_count == before - removed
    assert removed > 0


def test_reset_restores_full_model(make_env, stub_model):
    """reset() (Track1B) returns a fresh, unpruned model with full param count."""
    env = make_env(stub_model)
    run_episode(env)  # prune to completion
    assert env._param_count < env.original_params
    obs, _ = env.reset()
    assert env._param_count == env.original_params
    assert env._param_count == sum(p.numel() for p in env.model.parameters())
    assert obs[0] < 1.0  # not terminal

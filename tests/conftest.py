import numpy as np
import torch
import torch.nn as nn
import pytest


class StubAttention(nn.Module):
    """Minimal attention module matching the parts the env touches."""

    def __init__(self, num_heads=4, head_dim=3):
        super().__init__()
        self.self = nn.Module()
        self.self.num_attention_heads = num_heads
        self.self.attention_head_size = head_dim
        self.self.query = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.self.key = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.self.value = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.output = nn.Module()
        hidden = num_heads * head_dim
        self.output.dense = nn.Linear(hidden, hidden)
        self._hidden = hidden

    def prune_heads(self, heads_to_prune):
        n = len(heads_to_prune)
        if n == 0:
            return
        self.self.num_attention_heads -= n
        # shrink projections to match reduced head count
        old_hidden = self._hidden
        head_dim = self.self.attention_head_size
        new_hidden = self.self.num_attention_heads * head_dim
        for name in ("query", "key", "value"):
            lin = getattr(self.self, name)
            new_lin = nn.Linear(old_hidden, new_hidden)
            with torch.no_grad():
                new_lin.weight[:, :new_hidden] = lin.weight[:new_hidden, :new_hidden]
            setattr(self.self, name, new_lin)
        new_dense = nn.Linear(old_hidden, new_hidden)
        with torch.no_grad():
            new_dense.weight[:, :new_hidden] = self.output.dense.weight[:new_hidden, :new_hidden]
        self.output.dense = new_dense


class StubBertLayer(nn.Module):
    def __init__(self, hidden=8, ffn_dim=16, num_heads=4):
        super().__init__()
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden, ffn_dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(ffn_dim, hidden)
        self.attention = StubAttention(num_heads=num_heads, head_dim=hidden // num_heads)


class StubBertEncoder(nn.Module):
    def __init__(self, n_layers=3, hidden=8, ffn_dim=16, num_heads=4):
        super().__init__()
        self.layer = nn.ModuleList(
            [StubBertLayer(hidden, ffn_dim, num_heads) for _ in range(n_layers)]
        )


class StubBert(nn.Module):
    def __init__(self, n_layers=3, hidden=8, ffn_dim=16, num_heads=4):
        super().__init__()
        self.bert = nn.Module()
        self.bert.encoder = StubBertEncoder(n_layers, hidden, ffn_dim, num_heads)


@pytest.fixture
def stub_model():
    return StubBert(n_layers=3, hidden=8, ffn_dim=16, num_heads=4)


class StubEnv:
    """Drop-in stub env matching the interface train_ppo requires."""

    def __init__(self, n_layers=12, eval_acc=0.6, n_rollouts_any=0):
        import gymnasium as gym

        self.n_layers = n_layers
        self.action_space = gym.spaces.Box(
            low=0.0, high=0.8, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.original_params = 1000
        self.alpha = 0.05
        self._eval_acc = eval_acc
        self._layer = 0
        self.model = object()

    def reset(self, *, seed=None, options=None):
        self._layer = 0
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self._layer += 1
        done = self._layer >= self.n_layers
        rew = 0.5 if not done else 1.0
        return np.zeros(5, dtype=np.float32), rew, done, False, {}

    def eval_fn(self, model):
        return self._eval_acc

    def _count_parameters(self, model):
        return 800


@pytest.fixture
def stub_env():
    return StubEnv()

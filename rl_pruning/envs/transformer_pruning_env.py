from copy import deepcopy
import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
from rl_pruning.utils.priors import compute_layer_importance


class TransformerPruningEnv(gym.Env):
    def __init__(
        self,
        model,
        tokenizer,
        eval_fn,
        max_prune_ratio=0.8,
        alpha=0.3,
        acc_floor_ratio=0.85,
        shaping_coef=1.0,
        shaping_penalty=0.1,
        device="cuda",
    ):
        super().__init__()

        # Keep a pristine CPU template for cheap deepcopy in reset(); original_model
        # stays on-device for dimension lookups in _get_state().
        self._template = deepcopy(model)
        self.original_model = deepcopy(self._template).to(device)
        self.model = deepcopy(self._template).to(device)
        self.tokenizer = tokenizer
        self.eval_fn = eval_fn
        self.device = device

        self.layers = self.model.bert.encoder.layer
        self.n_layers = len(self.layers)
        self.max_prune_ratio = max_prune_ratio
        self.alpha = alpha
        self.acc_floor_ratio = acc_floor_ratio
        self.shaping_coef = shaping_coef
        self.shaping_penalty = shaping_penalty

        # Continuous action: prune ratio ∈ [0, max_prune_ratio]
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=max_prune_ratio,
            shape=(2,),
            dtype=np.float32,
        )

        # State = dynamic, per-layer only
        # [layer_idx, ffn_ratio, head_ratio, param_ratio, importance]
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        # compute full param count via direct scan before the running cache exists
        self._param_count = sum(p.numel() for p in self.model.parameters())
        self.original_params = self._param_count
        self.original_accuracy = self.eval_fn(self.model)

        # Layer importance priors
        self.layer_importance = compute_layer_importance(self.original_model)

        self.reset()

    # -------------------- Core RL API --------------------

    def reset(self, *, seed=None, options=None):
        self.model = deepcopy(self._template).to(self.device)
        self.layers = self.model.bert.encoder.layer
        self.current_layer = 0
        self.done = False
        self._param_count = self.original_params
        return self._get_state(), {}

    def step(self, action):
        assert not self.done

        layer = self.layers[self.current_layer]

        # ---- prune current layer (updates running _param_count) ----
        removed = self._prune_layer(layer, action)
        param_reduction = removed / self.original_params

        # ---- advance layer pointer ----
        layer_idx = self.current_layer
        self.current_layer += 1
        if self.current_layer >= self.n_layers:
            self.done = True

        # ---- reward ----
        if not self.done:
            # proxy reward (shaping signal; no eval). Aligned to the terminal
            # reward's compression term via shaping_coef.
            ffn_ratio, head_ratio = action
            importance = self.layer_importance[layer_idx]

            reward = self.shaping_coef * param_reduction - self.shaping_penalty * (
                importance * (ffn_ratio**2 + head_ratio**2)
            )
        else:
            # terminal reward: maximize accuracy retention while compressing
            accuracy = self.eval_fn(self.model)
            total_param_ratio = self._param_count / self.original_params
            compression = 1.0 - total_param_ratio

            retention = accuracy / self.original_accuracy
            reward = retention - self.alpha * compression
            # soft accuracy floor: strongly discourage catastrophic degradation
            if retention < self.acc_floor_ratio:
                reward -= 2.0

        return self._get_state(), float(reward), self.done, False, {}

    # -------------------- State --------------------

    def _get_state(self):
        idx = self.current_layer / self.n_layers if not self.done else 1.0

        if self.done:
            return np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    self._param_count / self.original_params,
                    0.0,
                ],
                dtype=np.float32,
            )

        layer = self.layers[self.current_layer]

        ffn_ratio = (
            layer.intermediate.dense.out_features
            / self.original_model.bert.encoder.layer[
                self.current_layer
            ].intermediate.dense.out_features
        )

        head_ratio = (
            layer.attention.self.num_attention_heads
            / self.original_model.bert.encoder.layer[
                self.current_layer
            ].attention.self.num_attention_heads
        )

        param_ratio = self._param_count / self.original_params
        importance = self.layer_importance[self.current_layer]

        return np.array(
            [idx, ffn_ratio, head_ratio, param_ratio, importance],
            dtype=np.float32,
        )

    # -------------------- Pruning --------------------

    def _prune_layer(self, layer, action):
        ffn_ratio = float(np.clip(action[0], 0.0, self.max_prune_ratio))
        head_ratio = float(np.clip(action[1], 0.0, self.max_prune_ratio))

        removed = self._prune_ffn(layer, ffn_ratio)
        removed += self._prune_heads(layer, head_ratio)
        self._param_count -= removed
        return removed

    def _prune_ffn(self, layer, ratio):
        fc1 = layer.intermediate.dense
        fc2 = layer.output.dense

        old_dim = fc1.out_features
        new_dim = int(old_dim * (1.0 - ratio))
        if new_dim < 1 or new_dim == old_dim:
            return 0

        # Exact removed param count for the FFN block
        removed = (old_dim - new_dim) * (fc1.in_features + 1 + fc2.out_features)

        # Importance-based neuron selection (L1 norm)
        scores = fc1.weight.abs().sum(dim=1)
        keep_idx = torch.topk(scores, k=new_dim, largest=True).indices

        new_fc1 = nn.Linear(fc1.in_features, new_dim, bias=True).to(self.device)
        new_fc2 = nn.Linear(new_dim, fc2.out_features, bias=True).to(self.device)

        new_fc1.weight.data = fc1.weight.data[keep_idx].clone()
        new_fc1.bias.data = fc1.bias.data[keep_idx].clone()

        new_fc2.weight.data = fc2.weight.data[:, keep_idx].clone()
        new_fc2.bias.data = fc2.bias.data.clone()

        layer.intermediate.dense = new_fc1
        layer.output.dense = new_fc2
        return removed

    def _count_attn_params(self, layer):
        attn = layer.attention
        total = 0
        for lin in (attn.self.query, attn.self.key, attn.self.value, attn.output.dense):
            total += lin.weight.numel()
            if lin.bias is not None:
                total += lin.bias.numel()
        return total

    def _prune_heads(self, layer, ratio):
        attn = layer.attention.self
        old_heads = attn.num_attention_heads
        new_heads = int(old_heads * (1.0 - ratio))

        if new_heads < 1 or new_heads == old_heads:
            return 0

        before = self._count_attn_params(layer)

        # Head importance w/ output projection weight norms
        head_dim = attn.attention_head_size
        W = layer.attention.output.dense.weight
        W = W.view(old_heads, head_dim, -1)
        scores = W.abs().sum(dim=(1, 2))

        keep = torch.topk(scores, k=new_heads).indices.tolist()
        heads_to_prune = [h for h in range(old_heads) if h not in keep]

        layer.attention.prune_heads(heads_to_prune)

        after = self._count_attn_params(layer)
        return before - after

    # -------------------- Utilities --------------------

    def _count_parameters(self, model=None):
        # When called with no arg (or the env's current model), return the
        # cached running count updated incrementally by _prune_layer.
        if model is None or model is self.model:
            return self._param_count
        return sum(p.numel() for p in model.parameters())


# --------------- Environment Calibration ---------------


def validate_environment(env, n_test_episodes=5):
    print("=== Environment Validation ===")

    param_counts = []
    accuracies = []
    rewards = []

    for ep in range(n_test_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        initial_params = env._count_parameters(env.model)

        while not done:
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        final_params = env._count_parameters(env.model)
        final_acc = env.eval_fn(env.model)

        param_counts.append(final_params)
        accuracies.append(final_acc)
        rewards.append(total_reward)

        print(
            f"Episode {ep}: Params={final_params:,} ({final_params / initial_params:.2%}), "
            f"Acc={final_acc:.3f}, Reward={total_reward:.4f}"
        )

    # Validate compression actually happens
    avg_compression = np.mean([p / env.original_params for p in param_counts])
    assert avg_compression < 0.99, "Environment not producing meaningful compression"
    assert np.mean(accuracies) > 0.3, "Accuracy dropped too drastically"

    print(f"✓ Environment validated: {avg_compression:.1%} avg compression")
    return param_counts, accuracies, rewards


# TODO: Fixed alphas...
def tune_alpha_parameter(env, alpha_candidates=[0.01, 0.05, 0.1, 0.2, 0.5]):
    print("=== Alpha Tuning ===")

    results = {}
    for alpha in alpha_candidates:
        env.alpha = alpha
        _, accs, _ = validate_environment(env, n_test_episodes=3)
        results[alpha] = np.mean(accs)
        print(f"α={alpha:.3f} → Avg Accuracy: {results[alpha]:.4f}")

    # Choose alpha that maintains >90% of original accuracy
    target_acc = 0.9 * env.original_accuracy
    best_alpha = min(
        [a for a, acc in results.items() if acc >= target_acc], default=0.05
    )

    print(f"✓ Selected α={best_alpha:.3f}")
    env.alpha = best_alpha
    return best_alpha

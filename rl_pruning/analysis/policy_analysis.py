import torch
import numpy as np


def analyze_pruning_policy(agent, env, device="cuda"):
    print("=== Policy Analysis ===")

    obs, _ = env.reset()
    done = False
    decisions = []  # (layer_idx, ffn_ratio, head_ratio)

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _, _ = agent.act(obs_t)

        layer_idx = env.current_layer
        # Handle 2D action [ffn_ratio, head_ratio]
        prune_ratios = action.cpu().numpy()
        ffn_ratio, head_ratio = float(prune_ratios[0]), float(prune_ratios[1])
        decisions.append((layer_idx, ffn_ratio, head_ratio))

        # Convert back to 1D for env.step (expects array)
        obs, _, done, _, _ = env.step(prune_ratios)

    # Sort by layer index
    decisions.sort(key=lambda x: x[0])

    print("\nLearned pruning ratios per layer:")
    print("Layer | FFN    | Heads  | Combined")
    print("------|--------|--------|---------")
    for layer, ffn_r, head_r in decisions:
        combined = (ffn_r + head_r) / 2
        print(f"{layer:5d} | {ffn_r:6.2%} | {head_r:6.2%} | {combined:7.2%}")

    # Calculate statistics
    ffn_ratios = [r for _, r, _ in decisions]
    head_ratios = [r for _, _, r in decisions]

    print(f"\nFFN pruning:  {np.mean(ffn_ratios):.3f} ± {np.std(ffn_ratios):.3f}")
    print(f"Head pruning: {np.mean(head_ratios):.3f} ± {np.std(head_ratios):.3f}")

    # Find extreme layers
    max_ffn = max(decisions, key=lambda x: x[1])
    max_head = max(decisions, key=lambda x: x[2])
    print(f"\nMost FFN-pruned:    Layer {max_ffn[0]} ({max_ffn[1]:.3f})")
    print(f"Most Head-pruned:   Layer {max_head[0]} ({max_head[2]:.3f})")

    return decisions

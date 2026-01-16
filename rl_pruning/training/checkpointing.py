from pathlib import Path
import torch


class CheckpointManager:
    def __init__(self, save_dir="./checkpoints", patience=20, min_delta=0.001):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -float("inf")
        self.wait_count = 0
        self.history = []

    def should_stop(self, reward):
        self.history.append(reward)

        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            return self.wait_count >= self.patience

    def save_checkpoint(self, net, optimizer, episode):
        checkpoint = {
            "episode": episode,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_reward": self.best_reward,
            "history": self.history,
        }
        path = self.save_dir / f"ppo_prune_ep{episode}_reward{self.best_reward:.4f}.pt"
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

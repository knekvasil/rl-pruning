from tqdm import tqdm
import time
import numpy as np
import torch
import torch.optim as optim
from .checkpointing import CheckpointManager
from rl_pruning.models.ppo import PPOActorCritic


def train_ppo(
    env,
    config,
    episodes: int,
    device: str,
    full_eval_fn=None,
    eval_every: int = 10,
):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Hyperparameters
    gamma = config.gamma
    lam = config.lam
    clip = config.clip
    entropy_coef = config.entropy_coef
    value_coef = config.value_coef
    lr = config.lr
    n_rollouts = config.n_rollouts

    net = PPOActorCritic(obs_dim, act_dim, max_action).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10
    )

    checkpoint_mgr = CheckpointManager(patience=75)  # 10, 75

    # Tracking
    episode_rewards = []
    episode_accuracies = []
    episode_compressions = []
    full_accuracies = []

    for episode in tqdm(range(episodes), desc="Training PPO"):
        ep_start = time.time()
        # Buffers accumulate transitions across all rollouts in this update
        obs_buf, act_buf, logp_buf = [], [], []
        adv_buf, ret_buf = [], []

        roll_rewards, roll_accs, roll_comps = [], [], []

        for _ in range(n_rollouts):
            obs, _ = env.reset()
            done = False

            # Per-rollout trajectory buffers
            r_obs, r_act, r_logp, r_rew, r_val = [], [], [], [], []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                action, u, logp, val = net.act(obs_t)

                next_obs, reward, done, _, _ = env.step(action.cpu().numpy())

                r_obs.append(obs_t)
                r_act.append(u.detach())
                r_logp.append(logp.detach())
                r_val.append(val.detach())
                r_rew.append(torch.tensor(reward, device=device))

                obs = next_obs

            # -------- GAE over this trajectory --------
            vals = r_val + [torch.tensor(0.0, device=device)]
            adv, gae = [], 0.0

            for t in reversed(range(len(r_rew))):
                # Optional monotonic layer weighting (0 disables)
                proxy_reward = r_rew[t] * (
                    1.0 + config.layer_weight_decay * t / len(r_rew)
                )
                delta = proxy_reward + gamma * vals[t + 1] - vals[t]
                gae = delta + gamma * lam * gae
                adv.insert(0, gae)

            adv_t = torch.stack(adv)
            ret_t = adv_t + torch.stack(r_val)

            # Accumulate into shared update batch
            obs_buf += r_obs
            act_buf += r_act
            logp_buf += r_logp
            adv_buf.append(adv_t)
            ret_buf.append(ret_t)

            # Per-rollout metrics
            roll_rewards.append(sum(r_rew).item())
            roll_accs.append(env.eval_fn(env.model))
            roll_comps.append(env._count_parameters(env.model) / env.original_params)

        adv = torch.cat(adv_buf)
        ret = torch.cat(ret_buf)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # -------- PPO Update --------
        obs_batch = torch.stack(obs_buf)
        act_batch = torch.stack(act_buf)
        logp_old = torch.stack(logp_buf).detach()

        # Multi-epoch policy update
        for epoch in range(config.epochs):
            logp, entropy, value = net.evaluate(obs_batch, act_batch)
            ratio = (logp - logp_old).exp()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (ret - value).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            opt.step()

        # -------- Logging & Monitoring (aggregate across rollouts) --------
        final_reward = float(np.mean(roll_rewards))
        final_accuracy = float(np.mean(roll_accs))
        final_compression = float(np.mean(roll_comps))

        full_accuracy = None
        if full_eval_fn is not None and episode % eval_every == 0:
            full_accuracy = full_eval_fn(env.model)
            full_accuracies.append((episode, full_accuracy))

        episode_rewards.append(final_reward)
        episode_accuracies.append(final_accuracy)
        episode_compressions.append(final_compression)

        # Learning rate scheduling
        scheduler.step(final_reward)

        # Early stopping check
        if checkpoint_mgr.should_stop(final_reward):
            print(f"\n⚠ Early stopping triggered at episode {episode}")
            break

        # Periodic checkpointing
        if episode > 0 and episode % 20 == 0:
            checkpoint_mgr.save_checkpoint(net, opt, episode)

        if episode % 10 == 0:
            acc_str = f"Acc: {final_accuracy:.4f}"
            if full_accuracy is not None:
                acc_str += f" (full: {full_accuracy:.4f})"
            print(
                f"\nEpisode {episode:3d} | "
                f"Reward: {final_reward:7.4f} | "
                f"{acc_str} | "
                f"Comp: {final_compression:.2%} | "
                f"LR: {opt.param_groups[0]['lr']:.1e} | "
                f"Time: {time.time() - ep_start:.1f}s"
            )

    # Save final model
    checkpoint_mgr.save_checkpoint(net, opt, episode)

    # Return training metrics
    history = {
        "rewards": episode_rewards,
        "accuracies": episode_accuracies,
        "compressions": episode_compressions,
        "full_accuracies": full_accuracies,
        "alpha": env.alpha,
    }

    return net, history

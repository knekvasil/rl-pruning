import torch
import torch.nn as nn


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super().__init__()
        self.max_action = max_action

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.mu(h), self.v(h)

    def act(self, obs):
        mu, v = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)

        action = torch.clamp(
            torch.tanh(action) * self.max_action,
            0.0,
            self.max_action,
        )
        return action, logp, v.squeeze(-1)

    def evaluate(self, obs, action):
        mu, v = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        # inverse tanh
        raw_action = torch.atanh(torch.clamp(action / self.max_action, -0.999, 0.999))
        logp = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return logp, entropy, v.squeeze(-1)

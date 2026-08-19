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

    def _log_jacobian(self, u):
        # a = 0.5 * max * (tanh(u) + 1)  =>  da/du = 0.5 * max * sech^2(u)
        # log|da/du| = log(max/2) - 2*log(cosh(u))
        # log(cosh(u)) = |u| + log1p(exp(-2|u|)) - log(2)  (stable for large u)
        log_cosh = u.abs() + torch.log1p(torch.exp(-2.0 * u.abs())) - torch.log(
            torch.tensor(2.0, device=u.device)
        )
        return (torch.log(torch.tensor(self.max_action / 2.0, device=u.device))
                - 2.0 * log_cosh).sum(-1)

    def act(self, obs):
        mu, v = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.sample()
        action = torch.clamp(
            0.5 * self.max_action * (torch.tanh(u) + 1.0), 0.0, self.max_action
        )
        logp = dist.log_prob(u).sum(-1) - self._log_jacobian(u)
        return action, u, logp, v.squeeze(-1)

    def evaluate(self, obs, u):
        mu, v = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        logp = dist.log_prob(u).sum(-1) - self._log_jacobian(u)
        entropy = dist.entropy().sum(-1)

        return logp, entropy, v.squeeze(-1)

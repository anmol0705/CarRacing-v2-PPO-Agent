"""ActorCritic CNN model for continuous CarRacing control."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Shared-backbone CNN with separate actor (Gaussian) and critic heads.

    Architecture:
        Conv2d(4,32,8,4) -> ReLU -> Conv2d(32,64,4,2) -> ReLU ->
        Conv2d(64,64,3,1) -> ReLU -> Flatten -> Linear(64*7*7, 512) -> ReLU
        -> mu_head (Linear 512->3) + log_std (learnable param)
        -> value_head (Linear 512->1)

    Actions are sampled from Normal(mu, std) and clamped to valid ranges.
    No tanh/sigmoid on the mean — this avoids gradient saturation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(512, 3)
        self.log_std = nn.Parameter(torch.zeros(3))
        self.value_head = nn.Linear(512, 1)

        # Action space: steer [-1,1], gas [0,1], brake [0,1]
        self.register_buffer("act_low", torch.tensor([-1.0, 0.0, 0.0]))
        self.register_buffer("act_high", torch.tensor([1.0, 1.0, 1.0]))
        # Center of action space for each dimension
        self.register_buffer("act_center", torch.tensor([0.0, 0.5, 0.5]))
        self.register_buffer("act_scale", torch.tensor([1.0, 0.5, 0.5]))

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization with appropriate gains."""
        for module in self.cnn:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _get_mu(self, features: torch.Tensor) -> torch.Tensor:
        """Get action mean, scaled to action space center."""
        raw = self.mu_head(features)
        # Scale raw output to action range center + scale
        return self.act_center + self.act_scale * torch.tanh(raw)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action, log_prob, value, entropy.

        Args:
            obs: Observation tensor of shape (batch, 4, 84, 84).

        Returns:
            action: Sampled action, shape (batch, 3).
            log_prob: Log probability of action, shape (batch,).
            value: State value estimate, shape (batch, 1).
            entropy: Distribution entropy, shape (batch,).
        """
        features = self.cnn(obs)

        mu = self._get_mu(features)
        std = self.log_std.clamp(-3, 0).exp()
        dist = Normal(mu, std)

        action = dist.sample()
        action = torch.max(torch.min(action, self.act_high), self.act_low)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.value_head(features)

        return action, log_prob, value, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return only the value estimate for given observations."""
        features = self.cnn(obs)
        return self.value_head(features)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions under current policy.

        Returns:
            log_prob: Log probability of actions, shape (batch,).
            value: State value estimate, shape (batch, 1).
            entropy: Distribution entropy, shape (batch,).
        """
        features = self.cnn(obs)

        mu = self._get_mu(features)
        std = self.log_std.clamp(-3, 0).exp()
        dist = Normal(mu, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(features)

        return log_prob, value, entropy

    def get_greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic (mean) action for evaluation."""
        features = self.cnn(obs)
        return self._get_mu(features)

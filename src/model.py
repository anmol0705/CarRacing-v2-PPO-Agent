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

    Action mapping:
        action[0] (steer):  tanh    -> [-1, 1]
        action[1] (gas):    sigmoid -> [0, 1]
        action[2] (brake):  sigmoid -> [0, 1]
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

    def _squash_actions(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw network output to valid action ranges.

        steer: tanh -> [-1, 1], gas: sigmoid -> [0, 1], brake: sigmoid -> [0, 1]
        """
        steer = torch.tanh(raw[:, 0:1])
        gas = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        return torch.cat([steer, gas, brake], dim=1)

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

        mu = self._squash_actions(self.mu_head(features))
        std = self.log_std.clamp(-2, 0.5).exp()
        dist = Normal(mu, std)

        action = dist.sample()
        # Clamp to valid ranges: steer [-1,1], gas [0,1], brake [0,1]
        action = torch.cat([
            action[:, 0:1].clamp(-1, 1),
            action[:, 1:2].clamp(0, 1),
            action[:, 2:3].clamp(0, 1),
        ], dim=1)
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

        mu = self._squash_actions(self.mu_head(features))
        std = self.log_std.clamp(-2, 0.5).exp()
        dist = Normal(mu, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(features)

        return log_prob, value, entropy

    def get_greedy_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic (mean) action for evaluation."""
        features = self.cnn(obs)
        return self._squash_actions(self.mu_head(features))

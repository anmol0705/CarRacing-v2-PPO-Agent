"""PPO trainer for highway-env scenarios (highway, roundabout, parking)."""

import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal


@dataclass
class HighwayPPOConfig:
    """All hyperparameters for highway-env PPO training."""

    env_name: str = "highway-v0"
    n_envs: int = 8
    total_timesteps: int = 500_000
    rollout_steps: int = 128
    n_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    checkpoint_freq: int = 50_000
    log_freq: int = 5_000
    eval_freq: int = 25_000
    eval_episodes: int = 10
    checkpoint_dir: str = "checkpoints/highway"
    log_file: str = "logs/highway_train.log"


def preprocess_obs(obs, env_name: str) -> np.ndarray:
    """Flatten highway-env observations to 1D float32 array."""
    if isinstance(obs, dict):
        parts = []
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key in obs:
                parts.append(np.array(obs[key]).flatten())
        return np.concatenate(parts).astype(np.float32)
    return np.array(obs).flatten().astype(np.float32)


ENV_CONFIGS = {
    "highway-v0": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,
        "vehicles_count": 10,
        "duration": 40,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 5,
        "policy_frequency": 5,
    },
    "roundabout-v0": {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": False,
        },
        "action": {"type": "DiscreteMetaAction"},
        "duration": 11,
        "simulation_frequency": 5,
        "policy_frequency": 5,
    },
    "parking-v0": {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "duration": 100,
        "simulation_frequency": 15,
    },
}


def make_highway_env(
    env_name: str, seed: int = 0, render: bool = False
) -> gym.Env:
    """Create a configured highway-env instance."""
    render_mode = "rgb_array" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    config = ENV_CONFIGS.get(env_name, {})
    if config:
        env.unwrapped.config.update(config)
        env.reset(seed=seed)
    return env


class HighwayActorCritic(nn.Module):
    """MLP actor-critic for vector observations (discrete or continuous)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        discrete: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action, log_prob, value, entropy."""
        features = self.shared(x)
        logits_or_mu = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)

        if self.discrete:
            dist = Categorical(logits=logits_or_mu)
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            std = self.log_std.exp().expand_as(logits_or_mu)
            dist = Normal(logits_or_mu, std)
            action = logits_or_mu if deterministic else dist.rsample()
            action = action.clamp(-1, 1)
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)

        return action, log_prob, value, entropy

    def evaluate_actions(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions under current policy."""
        features = self.shared(x)
        logits_or_mu = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)

        if self.discrete:
            dist = Categorical(logits=logits_or_mu)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            std = self.log_std.exp().expand_as(logits_or_mu)
            dist = Normal(logits_or_mu, std)
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

        return log_prob, value, entropy

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return value estimate only."""
        return self.critic_head(self.shared(x)).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


class HighwayPPOTrainer:
    """PPO trainer for highway-env vector-observation environments."""

    def __init__(self, cfg: HighwayPPOConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        self.envs = [make_highway_env(cfg.env_name, seed=i) for i in range(cfg.n_envs)]
        sample_obs, _ = self.envs[0].reset()
        self.obs_dim = preprocess_obs(sample_obs, cfg.env_name).shape[0]

        action_space = self.envs[0].action_space
        self.discrete = hasattr(action_space, "n")
        self.action_dim = action_space.n if self.discrete else action_space.shape[0]

        print(f"Env: {cfg.env_name}")
        print(f"Obs dim: {self.obs_dim}, Action dim: {self.action_dim}, Discrete: {self.discrete}")
        print(f"Device: {self.device}")

        self.model = HighwayActorCritic(
            self.obs_dim, self.action_dim, self.discrete
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, eps=1e-5)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {total_params:,}")

    def _preprocess_batch(self, obs_list: list) -> torch.Tensor:
        arr = np.array([preprocess_obs(o, self.cfg.env_name) for o in obs_list])
        return torch.FloatTensor(arr).to(self.device)

    def train(self) -> float:
        """Run full training loop. Returns best eval reward."""
        cfg = self.cfg
        log_f = open(cfg.log_file, "w", buffering=1)

        def log(msg: str) -> None:
            ts = time.strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            print(line)
            log_f.write(line + "\n")

        log(f"Training: {cfg.total_timesteps} steps, {cfg.n_envs} envs, env={cfg.env_name}")

        obs_list = [env.reset()[0] for env in self.envs]
        global_step = 0
        episode_rewards = [0.0] * cfg.n_envs
        episode_count = 0
        recent_rewards: list[float] = []
        best_eval = -np.inf
        n_updates = cfg.total_timesteps // (cfg.n_envs * cfg.rollout_steps)

        for update in range(1, n_updates + 1):
            frac = 1.0 - (update - 1) / n_updates
            lr = frac * cfg.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            b_obs, b_actions, b_logprobs = [], [], []
            b_rewards, b_dones, b_values = [], [], []

            for step in range(cfg.rollout_steps):
                global_step += cfg.n_envs
                obs_t = self._preprocess_batch(obs_list)

                with torch.no_grad():
                    actions, log_probs, values, _ = self.model.get_action(obs_t)

                actions_np = actions.cpu().numpy()
                b_obs.append(obs_t)
                b_actions.append(actions)
                b_logprobs.append(log_probs)
                b_values.append(values)

                rewards, dones, new_obs = [], [], []
                for i, env in enumerate(self.envs):
                    act = int(actions_np[i]) if self.discrete else actions_np[i]
                    obs, rew, term, trunc, _ = env.step(act)
                    done = term or trunc
                    episode_rewards[i] += rew
                    if done:
                        episode_count += 1
                        recent_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0.0
                        obs, _ = env.reset()
                    rewards.append(rew)
                    dones.append(float(done))
                    new_obs.append(obs)

                obs_list = new_obs
                b_rewards.append(torch.FloatTensor(rewards).to(self.device))
                b_dones.append(torch.FloatTensor(dones).to(self.device))

            with torch.no_grad():
                next_value = self.model.get_value(self._preprocess_batch(obs_list))

            rewards_t = torch.stack(b_rewards)
            values_t = torch.stack(b_values)
            dones_t = torch.stack(b_dones)
            advantages, returns = compute_gae(
                rewards_t, values_t, dones_t, next_value, cfg.gamma, cfg.gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            flat_obs = torch.stack(b_obs).reshape(-1, self.obs_dim)
            flat_act = torch.stack(b_actions).reshape(-1) if self.discrete else torch.stack(b_actions).reshape(-1, self.action_dim)
            flat_lp = torch.stack(b_logprobs).reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)
            batch_size = cfg.n_envs * cfg.rollout_steps

            for _epoch in range(cfg.n_epochs):
                idx = torch.randperm(batch_size, device=self.device)
                epoch_kl = 0.0
                n_mb = 0
                for start in range(0, batch_size, cfg.minibatch_size):
                    mb = idx[start : start + cfg.minibatch_size]
                    new_lp, new_val, new_ent = self.model.evaluate_actions(
                        flat_obs[mb], flat_act[mb]
                    )
                    ratio = (new_lp - flat_lp[mb]).exp()
                    pg1 = -flat_adv[mb] * ratio
                    pg2 = -flat_adv[mb] * ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
                    policy_loss = torch.max(pg1, pg2).mean()
                    value_loss = 0.5 * (new_val - flat_ret[mb]).pow(2).mean()
                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * new_ent.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (new_lp - flat_lp[mb])).mean().item()
                    epoch_kl += approx_kl
                    n_mb += 1

                if n_mb > 0 and epoch_kl / n_mb > cfg.target_kl:
                    break

            if global_step % cfg.log_freq < cfg.n_envs * cfg.rollout_steps:
                mean_r = np.mean(recent_rewards[-50:]) if recent_rewards else 0
                log(
                    f"Step {global_step:8d} | Ep {episode_count:5d} | "
                    f"MeanR {mean_r:8.2f} | LR {lr:.2e} | Loss {loss.item():.4f}"
                )

            if global_step % cfg.eval_freq < cfg.n_envs * cfg.rollout_steps:
                eval_r = self._evaluate()
                log(f"  EVAL @ {global_step}: {eval_r:.2f}")
                if eval_r > best_eval:
                    best_eval = eval_r
                    ckpt = f"{cfg.checkpoint_dir}/best.pt"
                    torch.save(
                        {"model": self.model.state_dict(), "step": global_step, "reward": eval_r},
                        ckpt,
                    )
                    log(f"  New best: {eval_r:.2f}")

            if global_step % cfg.checkpoint_freq < cfg.n_envs * cfg.rollout_steps:
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "step": global_step,
                        "reward": np.mean(recent_rewards[-20:]) if recent_rewards else 0,
                    },
                    f"{cfg.checkpoint_dir}/model_{global_step}.pt",
                )

        log(f"Training complete. Best eval: {best_eval:.2f}")
        log_f.close()
        for env in self.envs:
            env.close()
        return best_eval

    def _evaluate(self) -> float:
        """Run eval episodes with deterministic policy."""
        rewards = []
        env = make_highway_env(self.cfg.env_name)
        for ep in range(self.cfg.eval_episodes):
            obs, _ = env.reset(seed=ep + 1000)
            total_r = 0.0
            for _ in range(1000):
                obs_t = torch.FloatTensor(
                    preprocess_obs(obs, self.cfg.env_name)
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _, _ = self.model.get_action(obs_t, deterministic=True)
                act = int(action.item()) if self.discrete else action.cpu().numpy()[0]
                obs, r, term, trunc, _ = env.step(act)
                total_r += r
                if term or trunc:
                    break
            rewards.append(total_r)
        env.close()
        return float(np.mean(rewards))

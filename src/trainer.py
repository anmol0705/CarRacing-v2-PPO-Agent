"""Main PPO training loop for CarRacing."""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from omegaconf import DictConfig

from src.env_utils import make_vec_env
from src.evaluate import evaluate_policy
from src.model import ActorCritic
from src.ppo import compute_gae, ppo_update


class PPOTrainer:
    """Orchestrates PPO training: rollouts, updates, logging, checkpoints."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment (wrapped with episode statistics for reward tracking)
        self.envs = make_vec_env(
            n_envs=cfg.env.n_envs,
            seed=0,
        )
        self.envs = gym.wrappers.RecordEpisodeStatistics(self.envs)

        # Model and optimizer
        self.model = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.training.lr, eps=1e-5
        )

        # Rollout buffer shapes
        n_envs = cfg.env.n_envs
        n_steps = cfg.training.rollout_steps
        self.obs_buf = torch.zeros(n_steps, n_envs, 4, 84, 84, device=self.device)
        self.act_buf = torch.zeros(n_steps, n_envs, 3, device=self.device)
        self.logp_buf = torch.zeros(n_steps, n_envs, device=self.device)
        self.rew_buf = torch.zeros(n_steps, n_envs, device=self.device)
        self.done_buf = torch.zeros(n_steps, n_envs, device=self.device)
        self.val_buf = torch.zeros(n_steps, n_envs, device=self.device)

        # Tracking
        self.global_step = 0
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []

        # Directories
        Path("checkpoints").mkdir(exist_ok=True)
        Path("assets").mkdir(exist_ok=True)

    def _linear_lr(self) -> float:
        """Compute linearly decaying learning rate."""
        total = self.cfg.training.total_timesteps
        frac = 1.0 - self.global_step / total
        return max(frac * self.cfg.training.lr, 0.0)

    def _update_lr(self) -> float:
        """Apply linear LR schedule to optimizer."""
        lr = self._linear_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def collect_rollout(self, obs: np.ndarray) -> np.ndarray:
        """Collect rollout_steps of experience from vectorized envs.

        Args:
            obs: Current observation from envs, shape (n_envs, 4, 84, 84).

        Returns:
            Next observation after rollout.
        """
        n_steps = self.cfg.training.rollout_steps
        n_envs = self.cfg.env.n_envs

        for step in range(n_steps):
            obs_t = torch.as_tensor(
                np.array(obs), dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                action, log_prob, value, _ = self.model(obs_t)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = self.envs.step(action_np)
            done = np.logical_or(terminated, truncated)

            self.obs_buf[step] = obs_t
            self.act_buf[step] = action
            self.logp_buf[step] = log_prob
            self.rew_buf[step] = torch.as_tensor(reward, device=self.device)
            self.done_buf[step] = torch.as_tensor(
                done, dtype=torch.float32, device=self.device
            )
            self.val_buf[step] = value.squeeze(-1)

            # Track completed episodes
            if "_final_info" in infos:
                for i, final_info in enumerate(infos["_final_info"]):
                    if final_info is not None and "episode" in infos:
                        ep_r = float(infos["episode"]["r"][i])
                        ep_l = int(infos["episode"]["l"][i])
                        self.ep_rewards.append(ep_r)
                        self.ep_lengths.append(ep_l)

            obs = next_obs
            self.global_step += n_envs

        return obs

    def train(self) -> None:
        """Run the full PPO training loop."""
        cfg = self.cfg
        total_timesteps = cfg.training.total_timesteps
        n_envs = cfg.env.n_envs
        n_steps = cfg.training.rollout_steps
        batch_size = n_envs * n_steps

        # Initialize W&B
        run = wandb.init(
            project=cfg.logging.wandb_project,
            config=dict(cfg),
            save_code=False,
        )

        obs, _ = self.envs.reset()
        start_time = time.time()
        update_count = 0
        next_eval_step = cfg.logging.eval_interval
        next_ckpt_step = cfg.logging.checkpoint_interval
        next_gif_step = cfg.logging.gif_interval

        print(f"Starting training: {total_timesteps} steps, {n_envs} envs")
        print(f"Batch size: {batch_size}, minibatch: {cfg.training.minibatch_size}")

        while self.global_step < total_timesteps:
            lr = self._update_lr()
            obs = self.collect_rollout(obs)

            # Compute GAE
            with torch.no_grad():
                obs_t = torch.as_tensor(
                    np.array(obs), dtype=torch.float32, device=self.device
                )
                last_value = self.model.get_value(obs_t).squeeze(-1)

            advantages, returns = compute_gae(
                self.rew_buf,
                self.val_buf,
                self.done_buf,
                last_value,
                gamma=cfg.training.gamma,
                lam=cfg.training.gae_lambda,
            )

            # Flatten (T, n_envs, ...) -> (T*n_envs, ...)
            flat_obs = self.obs_buf.reshape(-1, 4, 84, 84)
            flat_act = self.act_buf.reshape(-1, 3)
            flat_logp = self.logp_buf.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret = returns.reshape(-1)

            # PPO update
            metrics = ppo_update(
                model=self.model,
                optimizer=self.optimizer,
                obs=flat_obs,
                actions=flat_act,
                old_log_probs=flat_logp,
                advantages=flat_adv,
                returns=flat_ret,
                clip_eps=cfg.training.clip_eps,
                vf_coef=cfg.training.vf_coef,
                ent_coef=cfg.training.ent_coef,
                max_grad_norm=cfg.training.max_grad_norm,
                target_kl=cfg.training.target_kl,
                n_epochs=cfg.training.n_epochs,
                minibatch_size=cfg.training.minibatch_size,
            )

            update_count += 1
            elapsed = time.time() - start_time
            sps = self.global_step / max(elapsed, 1)

            # Log to W&B
            log_dict = {
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/entropy": metrics["entropy"],
                "train/approx_kl": metrics["approx_kl"],
                "train/lr": lr,
                "train/sps": sps,
            }

            if self.ep_rewards:
                log_dict["train/ep_reward_mean"] = np.mean(self.ep_rewards)
                log_dict["train/ep_len_mean"] = np.mean(self.ep_lengths)
                print(
                    f"Step {self.global_step:>8d} | "
                    f"Reward: {np.mean(self.ep_rewards):>7.1f} | "
                    f"Policy: {metrics['policy_loss']:.4f} | "
                    f"Value: {metrics['value_loss']:.4f} | "
                    f"Ent: {metrics['entropy']:.4f} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"SPS: {sps:.0f}"
                )
                self.ep_rewards.clear()
                self.ep_lengths.clear()

            wandb.log(log_dict, step=self.global_step)

            # Evaluation
            if self.global_step >= next_eval_step:
                record = self.global_step >= next_gif_step
                gif_path = f"assets/step_{self.global_step}.gif" if record else ""
                mean_r, std_r = evaluate_policy(
                    self.model,
                    cfg,
                    n_episodes=cfg.logging.eval_episodes,
                    record_gif=record,
                    gif_path=gif_path,
                )
                wandb.log(
                    {"eval/mean_reward": mean_r, "eval/std_reward": std_r},
                    step=self.global_step,
                )
                print(f"  EVAL @ {self.global_step}: {mean_r:.1f} +/- {std_r:.1f}")
                next_eval_step += cfg.logging.eval_interval
                if record:
                    next_gif_step += cfg.logging.gif_interval

            # Checkpoint
            if self.global_step >= next_ckpt_step:
                ckpt_path = f"checkpoints/model_{self.global_step}.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "global_step": self.global_step,
                    },
                    ckpt_path,
                )
                print(f"  Checkpoint saved: {ckpt_path}")
                next_ckpt_step += cfg.logging.checkpoint_interval

        # Final checkpoint
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            "checkpoints/model_final.pt",
        )
        print(f"Training complete: {self.global_step} steps in {elapsed:.0f}s")

        self.envs.close()
        wandb.finish()

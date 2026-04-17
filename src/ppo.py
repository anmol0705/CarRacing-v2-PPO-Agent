"""PPO algorithm: GAE computation and clipped policy-value update."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ActorCritic


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Shape (T,) or (T, n_envs).
        values: Shape (T,) or (T, n_envs) — values for each timestep.
        dones: Shape (T,) or (T, n_envs) — 1.0 if episode ended.
        last_value: Shape () or (n_envs,) — bootstrap value.
        gamma: Discount factor.
        lam: GAE lambda for bias-variance tradeoff.

    Returns:
        advantages: Same shape as rewards.
        returns: Same shape as rewards (advantages + values).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float,
    n_epochs: int,
    minibatch_size: int,
) -> dict[str, float]:
    """Run PPO update epochs over collected rollout data.

    Args:
        model: ActorCritic network.
        optimizer: Optimizer for model parameters.
        obs: Flattened observations, shape (N, 4, 84, 84).
        actions: Flattened actions, shape (N, 3).
        old_log_probs: Flattened old log probs, shape (N,).
        old_values: Flattened old value predictions, shape (N,).
        advantages: Flattened advantages, shape (N,).
        returns: Flattened returns, shape (N,).
        clip_eps: PPO clipping epsilon.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.
        max_grad_norm: Max gradient norm for clipping.
        target_kl: Early stopping threshold for approximate KL.
        n_epochs: Number of optimization epochs.
        minibatch_size: Minibatch size for each gradient step.

    Returns:
        Dictionary of mean losses and metrics.
    """
    N = obs.shape[0]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_frac = 0.0
    n_updates = 0
    early_stopped = False

    for _epoch in range(n_epochs):
        # Random permutation for minibatches
        indices = torch.randperm(N, device=obs.device)

        for start in range(0, N, minibatch_size):
            end = min(start + minibatch_size, N)
            mb_idx = indices[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            new_log_probs, values, entropy = model.evaluate_actions(
                mb_obs, mb_actions
            )
            values = values.squeeze(-1)

            # Policy loss with clipping
            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (unclipped MSE — simpler and often better)
            value_loss = 0.5 * (values - mb_returns).pow(2).mean()

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                log_ratio = new_log_probs - mb_old_log_probs
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_approx_kl += approx_kl
            total_clip_frac += clip_frac
            n_updates += 1

        # Early stopping on KL divergence (check after each full epoch)
        if total_approx_kl / max(n_updates, 1) > target_kl:
            early_stopped = True
            break

    n_updates = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_approx_kl / n_updates,
        "clip_frac": total_clip_frac / n_updates,
        "early_stopped": float(early_stopped),
    }

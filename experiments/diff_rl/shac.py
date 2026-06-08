"""
Short-Horizon Actor-Critic (SHAC) agent for differentiable RL.
"""
import os
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')
from diff_rl.models import GaussianActor, EnsembleCritic
from diff_rl.utils import (
    CriticDataset, soft_update, grad_norm, policy_kl,
    adaptive_scheduler, RunningMeanStd, RewardShaper
)


class SHACAgent:
    """
    Short-Horizon Actor-Critic agent for differentiable simulation.

    Roll out short trajectories through differentiable physics,
    compute returns with value bootstrapping, and backpropagate through
    the simulation to update the policy.
    """
    def __init__(self, env, config):
        """
        Args:
            env: Genesis environment
            config: Dictionary with training configuration
        """
        self.env = env
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Debug mode
        self.debug = config.get('debug', False)

        # Environment specs
        self.num_envs = env.n_envs
        self.obs_dim = env._obs_dim  # Set by init_diff_rl_env()
        self.action_dim = env._act_dim  # Set by init_diff_rl_env()
        self.max_episode_length = env._horizon

        # SHAC hyperparameters
        self.horizon_len = config.get('horizon_len', 16)
        self.gamma = config.get('gamma', 0.99)
        self.critic_iterations = config.get('critic_iterations', 16)
        self.num_critic_batches = config.get('num_critic_batches', 4)
        self.critic_batch_size = self.num_envs * self.horizon_len // self.num_critic_batches
        self.no_target_critic = config.get('no_target_critic', False)  # Whether to disable target networks
        self.target_critic_alpha = config.get('target_critic_alpha', 0.4)
        self.critic_method = config.get('critic_method', 'one-step')  # 'one-step' or 'td-lambda'
        if self.critic_method == 'td-lambda':
            self.lam = config.get('lambda', 0.95)

        # Optimization
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.actor_lr = config.get('actor_lr', 1e-4)
        self.critic_lr = config.get('critic_lr', 1e-3)
        self.alpha_lr = config.get('alpha_lr', 5e-3)
        self.lr_schedule = config.get('lr_schedule', 'constant')  # 'constant', 'linear', 'kl'
        self.critic_lrschedule = config.get('critic_lrschedule', True)  # Whether to schedule critic LR
        self.min_lr = config.get('min_lr', 1e-6)
        self.max_lr = config.get('max_lr', self.actor_lr)
        self.max_epochs = config.get('max_epochs', 0)  # For linear schedule (0 = disabled)

        # Store initial learning rates for linear schedule
        self.actor_lr_init = self.actor_lr
        self.critic_lr_init = self.critic_lr
        self.last_lr = self.actor_lr  # For KL scheduler

        # Entropy regularization (SAC-style)
        self.with_entropy = config.get('with_entropy', False)
        self.with_logprobs = config.get('with_logprobs', False)
        self.entropy_coef = config.get('entropy_coef', None)  # Fixed entropy coef (alternative to auto-tuning)
        self.use_distr_ent = config.get('use_distr_ent', False)  # Use distribution entropy vs -logprob

        # Entropy scaling/offsetting options
        self.scale_by_target_entropy = config.get('scale_by_target_entropy', False)
        self.offset_by_target_entropy = config.get('offset_by_target_entropy', False)
        self.unscale_entropy_alpha = config.get('unscale_entropy_alpha', False)

        # Entropy in returns and targets
        self.entropy_in_return = config.get('entropy_in_return', False)
        self.entropy_in_targets = config.get('entropy_in_targets', False)
        self.no_actor_entropy = config.get('no_actor_entropy', False)

        # Target entropy
        self.target_entropy = -self.action_dim * config.get('target_entropy_scalar', 1.0)
        self.init_alpha = config.get('init_alpha', 0.1)

        # Observation normalization
        self.normalize_obs = config.get('normalize_obs', False)
        if self.normalize_obs:
            self.obs_rms = RunningMeanStd((self.obs_dim,))
        else:
            self.obs_rms = None

        # Reward shaping
        reward_scale = config.get('reward_scale', 1.0)
        reward_shift = config.get('reward_shift', 0.0)
        self.reward_shaper = RewardShaper(scale=reward_scale, shift=reward_shift)

        # Build networks
        self._build_networks()

        # Create buffers
        self._create_buffers()

        # Episode tracking
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.episode_rewards_hist = []
        self.episode_lengths_hist = []

        # Training state
        self.epoch = 0
        self.agent_steps = 0
        self.max_agent_steps = config.get('max_agent_steps', 1000000)
        self.avg_kl = None

        print(f"SHAC Agent initialized:")
        print(f"  Horizon: {self.horizon_len}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Critic batch size: {self.critic_batch_size}")
        print(f"  Actor LR: {self.actor_lr}, Critic LR: {self.critic_lr}")

    def _build_networks(self):
        """Build actor, critic, and target networks."""
        actor_hidden = self.config.get('actor_hidden_dims', (256, 256))
        critic_hidden = self.config.get('critic_hidden_dims', (256, 256))
        num_critics = self.config.get('num_critics', 2)
        activation = self.config.get('activation', 'relu')
        norm_type = self.config.get('norm_type', None)
        dist_kwargs = self.config.get('dist_kwargs', None)

        self.actor = GaussianActor(
            self.obs_dim, self.action_dim,
            hidden_dims=actor_hidden,
            activation=activation,
            norm_type=norm_type,
            dist_kwargs=dist_kwargs,
        ).to(self.device)

        self.critic = EnsembleCritic(
            self.obs_dim, self.action_dim,
            hidden_dims=critic_hidden,
            num_critics=num_critics,
            activation=activation,
            norm_type=norm_type,
        ).to(self.device)

        # Target critic for stable value learning (optional)
        if not self.no_target_critic:
            self.critic_target = deepcopy(self.critic)
        else:
            self.critic_target = self.critic

        # Optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, betas=(0.7, 0.95))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, betas=(0.7, 0.95))

        # Entropy coefficient (learnable if with_entropy)
        if self.with_entropy:
            self.log_alpha = nn.Parameter(torch.tensor(np.log(self.init_alpha), device=self.device))
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr, betas=(0.7, 0.95))

        self._debug_print(f"Actor: {self.actor}")
        self._debug_print(f"Critic: {self.critic}")

    def _debug_print(self, *args, **kwargs):
        """Print only when debug mode is enabled."""
        if self.debug:
            print(*args, **kwargs)

    def _create_buffers(self):
        """Create replay buffers for trajectory collection."""
        T, B = self.horizon_len, self.num_envs

        self.obs_buf = torch.zeros((T, B, self.obs_dim), dtype=torch.float32, device=self.device)
        self.action_buf = torch.zeros((T, B, self.action_dim), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((T, B), dtype=torch.float32, device=self.device)

        # For KL divergence computation
        self.mus = torch.zeros((T, B, self.action_dim), dtype=torch.float32, device=self.device)
        self.sigmas = torch.zeros((T, B, self.action_dim), dtype=torch.float32, device=self.device)

        # For log probabilities and distribution entropy (used with entropy regularization)
        if self.with_logprobs:
            self.logprobs = torch.zeros((T, B), dtype=torch.float32, device=self.device)
            self.distr_ent = torch.zeros((T, B), dtype=torch.float32, device=self.device)

    def get_action(self, obs, deterministic=False):
        """
        Get action from policy.

        Args:
            obs: Observation tensor (batch, obs_dim)
            deterministic: If True, return mean action

        Returns:
            action: Action tensor (batch, action_dim)
        """
        with torch.no_grad():
            if self.normalize_obs and self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            mu, sigma, distr = self.actor(obs)
            if deterministic:
                action = mu
            else:
                action = distr.rsample()

        return action

    def train_one_epoch(self):
        """
        Train for one epoch (one full episode across all environments).
        Each episode consists of multiple horizons.
        """
        self.epoch += 1

        # Initialize episode
        obs = self.env.initialize_trajectory()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Track which environments are still alive
        env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Number of horizons in one episode
        num_horizons = (self.max_episode_length + self.horizon_len - 1) // self.horizon_len

        epoch_metrics = defaultdict(list)

        for horizon_idx in range(num_horizons):
            # Check if all environments have failed
            if env_mask.sum() == 0:
                print(f"All environments failed at horizon {horizon_idx}/{num_horizons}. "
                      f"Ending epoch early (next epoch will reset and start fresh).")
                break

            # Cut gradients from previous horizon (but keep physical state!)
            obs = obs.detach()

            # Update learning rate if scheduled
            self._update_learning_rate()

            # Train actor for this horizon
            self.actor.train()
            self.critic.eval()
            self.critic_target.eval()
            obs, env_mask, actor_results = self.update_actor_one_horizon(obs, env_mask)
            self._debug_print(f"Actor update complete for horizon {horizon_idx}")

            # Compute target values for critic
            self._debug_print("Computing target values for critic...")
            with torch.no_grad():
                if self.entropy_in_targets:
                    self.compute_target_values_with_entropy()
                else:
                    self.compute_target_values()
            self._debug_print("Target values computed")

            # Train critic
            self._debug_print("Training critic...")
            self.actor.eval()
            self.critic.train()
            self._debug_print("Creating critic dataset...")
            dataset = CriticDataset(
                self.critic_batch_size,
                {'obs': self.obs_buf},
                self.target_values,
                drop_last=False
            )
            self._debug_print(f"Dataset created with {len(dataset)} batches")
            self._debug_print("Calling update_critic...")
            critic_results = self.update_critic(dataset)
            self._debug_print("Critic update complete")

            # Update target critic (if using target networks)
            if not self.no_target_critic:
                with torch.no_grad():
                    soft_update(self.critic, self.critic_target, self.target_critic_alpha)

            # Accumulate metrics
            for k, v in actor_results.items():
                epoch_metrics[k].extend(v)
            for k, v in critic_results.items():
                epoch_metrics[k].extend(v)

        return epoch_metrics

    def train(self):
        """Main training loop."""
        print("\nStarting SHAC training...")

        while self.agent_steps < self.max_agent_steps:

            # Train one full episode (multiple horizons)
            metrics = self.train_one_epoch()

            # Log metrics
            self._log_metrics(metrics)

            if self.epoch % 10 == 0:
                self._print_progress(metrics)

        print("\nTraining completed!")
        return self.episode_rewards_hist, self.episode_lengths_hist

    def update_actor_one_horizon(self, initial_obs, env_mask):
        """
        Update actor by rolling out ONE horizon through differentiable simulation.

        Args:
            initial_obs: Initial observation for this horizon (n_envs, obs_dim)
            env_mask: Binary mask indicating which environments are alive (n_envs,)

        Returns:
            final_obs: Final observation after this horizon
            final_env_mask: Updated environment mask
            results: Dictionary of metrics
        """
        results = defaultdict(list)
        obs = initial_obs

        # Zero out buffers
        with torch.no_grad():
            self.action_buf.zero_()
            self.mus.zero_()
            self.sigmas.zero_()
            if self.with_logprobs:
                self.logprobs.zero_()
                self.distr_ent.zero_()

        # Actor optimization step
        self.actor_optim.zero_grad()

        # Create fresh list for actions WITH gradients (for actor backprop)
        # This avoids reusing tensors from previous horizon's freed graph
        actions_with_grad = []

        # Compute actor loss through differentiable simulation
        final_obs, final_env_mask, returns, logprobs, distr_ents, horizon_idx_buffer = self.compute_actor_loss(obs, env_mask, actions_with_grad)

        # Mask out failed environments
        active_returns = returns[env_mask]

        if active_returns.numel() == 0:
            # No active environments - skip this update
            print("Warning: No active environments in this horizon. Skipping actor update.")
            return final_obs, final_env_mask, results

        # Normalize returns by horizon length
        active_returns = active_returns / self.horizon_len

        # Compute actor loss with different entropy handling modes
        if self.entropy_in_return or self.no_actor_entropy:
            # Entropy already included in returns, or explicitly disabled
            actor_loss = -active_returns.mean()
        elif (self.with_entropy or self.entropy_coef is not None) and self.with_logprobs:
            # Add entropy term separately (not discounted like in returns)
            alpha = self.get_alpha(scalar=True)
            entropy = distr_ents / self.horizon_len if self.use_distr_ent else -1.0 * logprobs / self.horizon_len

            # Apply scaling/offsetting
            if self.offset_by_target_entropy:
                entropy = (entropy + abs(self.target_entropy)) * 0.5
            if self.scale_by_target_entropy:
                entropy = entropy * (1.0 / abs(self.target_entropy))

            actor_loss = ((alpha * -entropy) - active_returns).mean()
            results['entropy'].append(entropy.mean().detach())
        else:
            # No entropy regularization
            actor_loss = -active_returns.mean()

        # Backpropagate through actor_loss
        # This populates ALL rope state gradients: ∂actor_loss/∂rope_verts[t] for all t
        self._debug_print(f'**** Backpropagating actor loss: {actor_loss.item()} ****')
        actor_loss.backward()  # Populates rope gradients through entire horizon!

        self._debug_print("Extracting rope vertex gradients (already computed by actor_loss.backward())...")

        # For each timestep, extract rope gradients and compute action gradients
        action_grads_list = []
        for step_idx in range(len(horizon_idx_buffer)):
            h_idx = horizon_idx_buffer[step_idx]

            # Extract gradients from grasped vertices
            # These are ∂actor_loss/∂rope_verts[t] (already includes full chain rule!)
            if self.env.task == 'separation':
                grasped_grad_1 = self.env.rope._queried_states[h_idx][0].pos.grad[:, self.env.control_idx[0], :]
                grasped_grad_2 = self.env.rope2._queried_states[h_idx][0].pos.grad[:, self.env.control_idx[1], :]
                grasped_grad = torch.stack([grasped_grad_1, grasped_grad_2], dim=1)
            else:
                grasped_grad = self.env.rope._queried_states[h_idx][0].pos.grad[:, self.env.control_idx, :]

            # Clean up gradients
            grasped_grad = torch.where(torch.isnan(grasped_grad), torch.zeros_like(grasped_grad), grasped_grad)
            grasped_grad = torch.where(torch.isinf(grasped_grad), torch.zeros_like(grasped_grad), grasped_grad)

            # Clip gradient norm (same as TrajOptimController)
            max_grad_norm = self.config.get('max_grad_norm', 1000.0)
            grad_norm_per_grasp = torch.linalg.norm(grasped_grad, dim=-1, keepdim=True)
            weight = max_grad_norm / (grad_norm_per_grasp + 1e-8)
            weight = torch.clamp(weight, max=1.0)
            grasped_grad = grasped_grad * weight

            # Reshape to action shape - these are ∂actor_loss/∂action[t] directly!
            # No need for chain rule multiplication - actor_loss.backward() already did it!
            action_grad_t = grasped_grad.reshape(self.num_envs, -1)  # (n_envs, action_dim)
            action_grads_list.append(action_grad_t)

        # Stack action gradients: shape (horizon_len, n_envs, action_dim)
        action_grads = torch.stack(action_grads_list, dim=0)

        # Backprop from actions to policy parameters
        # Use the fresh actions_with_grad list (not the reused buffer!)
        # This avoids backpropping through freed computational graphs from previous horizons
        actions_for_backprop = torch.stack(actions_with_grad)  # (horizon_len, n_envs, action_dim)
        actions_for_backprop.backward(action_grads)  # No retain_graph - final backward!

        # Clear queried states and gradients from simulator to prevent memory leak
        # This is crucial between horizons to avoid accumulating states across multiple rollouts
        self.env.scene.sim.reset_grad()

        # Reset scene state: Genesis's _backward() was triggered during actor_loss.backward()
        # which ran backward-through-time and corrupted the scene state
        # We need to restore it for the next horizon
        self.env.scene._forward_ready = True
        self.env.scene._backward_ready = True
        # Note: scene._t was decremented to 0 by _backward(), but that's OK for next horizon
        self._debug_print(f'Reset scene state - forward: {self.env.scene._forward_ready}, backward: {self.env.scene._backward_ready}, t: {self.env.scene._t}')

        # Gradient clipping
        self._debug_print("Computing gradient norms...")
        grad_norm_before = grad_norm(self.actor.parameters())
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        grad_norm_after = grad_norm(self.actor.parameters())

        self._debug_print(f"Applying actor optimizer step...")
        self.actor_optim.step()
        self._debug_print(f"Actor step complete. Grad norm: {grad_norm_before:.6f} → {grad_norm_after:.6f}")

        # Compute policy KL for adaptive LR scheduling
        self._debug_print("Computing policy KL divergence...")
        with torch.no_grad():
            obs = self.obs_buf.view(-1, self.obs_dim)
            if self.normalize_obs and self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            mu, sigma, _ = self.actor(obs, deterministic=False)
            old_mu = self.mus.view(-1, self.action_dim)
            old_sigma = self.sigmas.view(-1, self.action_dim)

            kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)
            avg_kl = kl_dist.mean() / self.action_dim
            self.avg_kl = avg_kl

        # Update entropy coefficient if using auto-entropy (learnable alpha)
        if self.with_entropy and self.entropy_coef is None and self.with_logprobs:
            entropy_for_alpha = distr_ents if self.use_distr_ent else -1.0 * logprobs
            self._update_alpha(entropy_for_alpha)

        self._debug_print("Storing results and returning...")
        results['actor_loss'].append(actor_loss.detach())
        results['returns'].append(active_returns.mean().detach())
        results['grad_norm_before'].append(grad_norm_before)
        results['grad_norm_after'].append(grad_norm_after)
        results['avg_kl'].append(avg_kl)
        results['num_active_envs'].append(env_mask.sum().item())

        self._debug_print(f"Returning from update_actor_one_horizon")
        return final_obs, final_env_mask, results

    def compute_actor_loss(self, initial_obs, env_mask, actions_with_grad):
        """
        Roll out ONE horizon through differentiable simulation and compute returns.

        Args:
            initial_obs: Initial observation (n_envs, obs_dim)
            env_mask: Binary mask for active environments (n_envs,)
            actions_with_grad: List to store actions WITH gradients (fresh for each horizon)

        Returns:
            final_obs: Final observation after horizon
            final_env_mask: Updated environment mask
            returns: Cumulative discounted returns (num_envs,)
            logprobs: Log probabilities (num_envs,)
            distr_ents: Distribution entropies (num_envs,)
        """
        obs = initial_obs

        if self.normalize_obs and self.obs_rms is not None:
            with torch.no_grad():
                self.obs_rms.update(obs)
            obs = self.obs_rms.normalize(obs)

        # Accumulators
        returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        logprobs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        distr_ents = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        rew_acc = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Store horizon_idx for manual gradient extraction from rope states
        horizon_idx_buffer = []

        for i in range(self.horizon_len):
            self._debug_print(f"Step {i+1}/{self.horizon_len} in horizon...")
            # Store observation
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            # Get action from policy
            mu, sigma, distr = self.actor(obs)
            action = distr.rsample()

            # Store action WITH gradients in fresh list (for this horizon's backprop)
            actions_with_grad.append(action)

            # Store action WITHOUT gradients in buffer (for critic training later)
            with torch.no_grad():
                self.action_buf[i] = action.clone()

            # Store mu/sigma for KL computation (no gradients needed)
            with torch.no_grad():
                self.mus[i] = mu.clone()
                self.sigmas[i] = sigma.clone()

            # Compute and store log probabilities and distribution entropy
            if self.with_logprobs:
                logprob = distr.log_prob(action).sum(dim=-1)
                distr_ent = distr.entropy().sum(dim=-1)
                with torch.no_grad():
                    self.logprobs[i] = logprob.clone()
                    self.distr_ent[i] = distr_ent.clone()
            else:
                logprob = None
                distr_ent = None

            # Apply action to environment and get loss
            loss, env_mask, raw_rew, horizon_idx = self.env.step_diff_rl(
                env_mask, action.detach()  # Detach for step_diff_rl (can convert to numpy)
            )

            # Store horizon_idx for manual gradient extraction later
            horizon_idx_buffer.append(horizon_idx)

            # Get new observation
            obs = self.env.compute_observation()
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            # Convert loss to reward for actor update (negative loss = reward)
            # loss shape: (n_envs,) - has gradients where env_mask=True
            rew = -loss  # Shape: (n_envs,)

            # Apply reward shaping
            rew = self.reward_shaper(rew)

            # Update episode metrics
            with torch.no_grad():
                self.episode_rewards += raw_rew
                self.episode_lengths += 1

            # Update observation normalization
            if self.normalize_obs and self.obs_rms is not None:
                with torch.no_grad():
                    self.obs_rms.update(obs)
                obs = self.obs_rms.normalize(obs)

            # Bootstrap value for next state
            # In differentiable RL, we need gradients through bootstrap values!
            # (Unlike standard RL where bootstrap is treated as fixed target)
            pred_val, _ = self.critic_target(obs, return_type='min_and_avg')
            pred_val = pred_val.squeeze(-1)

            # Store for critic training (detached copy)
            with torch.no_grad():
                self.next_values[i] = pred_val.clone()

            # Check which environments have failed (env_mask changed)
            # Failed environments get 0 bootstrap value
            failed_env_ids = (~env_mask).nonzero(as_tuple=False).squeeze(-1)
            if len(failed_env_ids) > 0:
                pred_val[failed_env_ids] = 0.0  # Zero out failed envs (WITH gradients)

            # Accumulate rewards (optionally with entropy term)
            if self.entropy_in_return and self.with_logprobs:
                # Add entropy term to reward (SAC-style)
                entropy = distr_ent.detach() if self.use_distr_ent else -1.0 * logprob.detach()
                if self.offset_by_target_entropy:
                    entropy = (entropy + abs(self.target_entropy)) * 0.5
                if self.scale_by_target_entropy:
                    entropy = entropy * (1.0 / abs(self.target_entropy))
                alpha = self.get_alpha(scalar=True)
                rew_acc = rew_acc + gamma * (rew + alpha * entropy)
            else:
                rew_acc = rew_acc + gamma * rew

            # Compute returns (use pred_val which HAS gradients, not self.next_values!)
            if i < self.horizon_len - 1:
                # Bootstrap for failed environments only
                if len(failed_env_ids) > 0:
                    rets = rew_acc[failed_env_ids] + self.gamma * gamma[failed_env_ids] * pred_val[failed_env_ids]
                    returns[failed_env_ids] += rets
            else:
                # Terminal step: bootstrap for all environments
                rets = rew_acc + self.gamma * gamma * pred_val
                returns += rets

            # Accumulate logprobs and entropies (for actor loss and alpha update)
            if self.with_logprobs:
                logprobs += logprob
                distr_ents += distr_ent

            # Update gamma
            gamma = gamma * self.gamma

            # Reset accumulators for failed environments
            if len(failed_env_ids) > 0:
                gamma[failed_env_ids] = 1.0
                rew_acc[failed_env_ids] = 0.0

            # Store for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                # Mark failed environments as done
                self.done_mask[i] = (~env_mask).float()

                # Track episodes for failed environments
                if len(failed_env_ids) > 0:
                    for env_id in failed_env_ids:
                        self.episode_rewards_hist.append(self.episode_rewards[env_id].item())
                        self.episode_lengths_hist.append(self.episode_lengths[env_id].item())
                        self.episode_rewards[env_id] = 0.0
                        self.episode_lengths[env_id] = 0

        self.agent_steps += self.horizon_len * self.num_envs

        # Return horizon_idx for manual gradient extraction from rope states
        return obs, env_mask, returns, logprobs, distr_ents, horizon_idx_buffer

    def update_critic(self, dataset):
        """
        Update critic to match target values.
        """
        results = defaultdict(list)

        for iter_idx in range(self.critic_iterations):
            self._debug_print(f"  Critic iteration {iter_idx+1}/{self.critic_iterations}")
            total_loss = 0.0
            # Manually iterate using indices (dataset has __getitem__ but no __iter__)
            for batch_idx in range(len(dataset)):
                self._debug_print(f"    Batch {batch_idx+1}/{len(dataset)}")
                obs_batch, target_batch = dataset[batch_idx]

                self.critic_optim.zero_grad()
                self._debug_print(f"      Computing critic loss...")
                critic_loss = self.compute_critic_loss(obs_batch['obs'], target_batch)
                self._debug_print(f"      Backward pass...")
                critic_loss.backward()

                self._debug_print(f"      Cleaning NaN gradients...")
                # Handle NaNs
                for param in self.critic.parameters():
                    if param.grad is not None:
                        param.grad.nan_to_num_(0.0, 0.0, 0.0)

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self._debug_print(f"      Optimizer step...")
                self.critic_optim.step()
                total_loss += critic_loss.item()
                self._debug_print(f"      Batch complete. Loss: {critic_loss.item():.6f}")

            avg_loss = total_loss / len(dataset)
            results['critic_loss'].append(avg_loss)

        return results

    def compute_critic_loss(self, obs, target_values):
        """
        Compute critic loss (MSE between predicted and target values).
        """
        # Debug: Check input observations
        self._debug_print(f"      [DEBUG] Target values - shape: {target_values.shape}, "
              f"min: {target_values.min().item():.4f}, max: {target_values.max().item():.4f}, "
              f"mean: {target_values.mean().item():.4f}, has_nan: {torch.isnan(target_values).any().item()}, "
              f"has_inf: {torch.isinf(target_values).any().item()}")

        # Check observations before normalization
        # obs can be either a tensor or a dict depending on context
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, torch.Tensor):
                    self._debug_print(f"      [DEBUG] obs['{k}'] - shape: {v.shape}, "
                          f"min: {v.min().item():.4f}, max: {v.max().item():.4f}, "
                          f"mean: {v.mean().item():.4f}, has_nan: {torch.isnan(v).any().item()}, "
                          f"has_inf: {torch.isinf(v).any().item()}")
        else:
            # obs is a tensor
            self._debug_print(f"      [DEBUG] obs - shape: {obs.shape}, "
                  f"min: {obs.min().item():.4f}, max: {obs.max().item():.4f}, "
                  f"mean: {obs.mean().item():.4f}, has_nan: {torch.isnan(obs).any().item()}, "
                  f"has_inf: {torch.isinf(obs).any().item()}")

        if self.normalize_obs and self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
            # Check after normalization
            self._debug_print(f"      [DEBUG] After normalization:")
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, torch.Tensor):
                        self._debug_print(f"      [DEBUG]   obs['{k}'] - min: {v.min().item():.4f}, "
                              f"max: {v.max().item():.4f}, mean: {v.mean().item():.4f}, "
                              f"has_nan: {torch.isnan(v).any().item()}, has_inf: {torch.isinf(v).any().item()}")
            else:
                self._debug_print(f"      [DEBUG]   obs - min: {obs.min().item():.4f}, "
                      f"max: {obs.max().item():.4f}, mean: {obs.mean().item():.4f}, "
                      f"has_nan: {torch.isnan(obs).any().item()}, has_inf: {torch.isinf(obs).any().item()}")

        pred_values = self.critic(obs, return_type='all')

        # Debug: Check predicted values
        self._debug_print(f"      [DEBUG] Number of critic heads: {len(pred_values)}")
        for i, pred in enumerate(pred_values):
            pred_squeezed = pred.squeeze(-1)
            self._debug_print(f"      [DEBUG] Critic head {i} - shape: {pred_squeezed.shape}, "
                  f"min: {pred_squeezed.min().item():.4f}, max: {pred_squeezed.max().item():.4f}, "
                  f"mean: {pred_squeezed.mean().item():.4f}, has_nan: {torch.isnan(pred_squeezed).any().item()}, "
                  f"has_inf: {torch.isinf(pred_squeezed).any().item()}")

        losses = [F.mse_loss(pred.squeeze(-1), target_values) for pred in pred_values]

        # Debug: Check individual losses
        self._debug_print(f"      [DEBUG] Individual losses: {[loss.item() for loss in losses]}")

        total_loss = torch.stack(losses).mean()
        self._debug_print(f"      [DEBUG] Total critic loss: {total_loss.item()}")

        return total_loss

    def compute_target_values(self):
        """
        Compute target values for critic training.
        """
        self._debug_print(f"  [DEBUG] Computing target values using method: {self.critic_method}")
        self._debug_print(f"  [DEBUG] rew_buf - shape: {self.rew_buf.shape}, "
              f"min: {self.rew_buf.min().item():.4f}, max: {self.rew_buf.max().item():.4f}, "
              f"mean: {self.rew_buf.mean().item():.4f}, has_nan: {torch.isnan(self.rew_buf).any().item()}")
        self._debug_print(f"  [DEBUG] next_values - shape: {self.next_values.shape}, "
              f"min: {self.next_values.min().item():.4f}, max: {self.next_values.max().item():.4f}, "
              f"mean: {self.next_values.mean().item():.4f}, has_nan: {torch.isnan(self.next_values).any().item()}")

        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
            self._debug_print(f"  [DEBUG] target_values (one-step) - "
                  f"min: {self.target_values.min().item():.4f}, max: {self.target_values.max().item():.4f}, "
                  f"mean: {self.target_values.mean().item():.4f}, has_nan: {torch.isnan(self.target_values).any().item()}")
        elif self.critic_method == 'td-lambda':
            # TD-lambda for multi-step returns
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)

            for i in reversed(range(self.horizon_len)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                adjusted_rew = (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + adjusted_rew)
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi

            self._debug_print(f"  [DEBUG] target_values (td-lambda) - "
                  f"min: {self.target_values.min().item():.4f}, max: {self.target_values.max().item():.4f}, "
                  f"mean: {self.target_values.mean().item():.4f}, has_nan: {torch.isnan(self.target_values).any().item()}")
        else:
            raise NotImplementedError(f"Unknown critic method: {self.critic_method}")

    def compute_target_values_with_entropy(self):
        """
        Compute target values with entropy term included (for SAC-style training).
        """
        self._debug_print(f"  [DEBUG] Computing target values WITH entropy using method: {self.critic_method}")

        # Get entropy term
        entropy = self.distr_ent if self.use_distr_ent else -1.0 * self.logprobs
        if self.offset_by_target_entropy:
            entropy = (entropy + abs(self.target_entropy)) * 0.5
        if self.scale_by_target_entropy:
            entropy = entropy * (1.0 / abs(self.target_entropy))

        alpha = self.get_alpha(scalar=True)

        if self.critic_method == 'one-step':
            self.target_values = (self.rew_buf + alpha * entropy) + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.horizon_len)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                rew = self.rew_buf[i] + alpha * entropy[i]
                adjusted_rew = (1.0 - lam) / (1.0 - self.lam) * rew
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + adjusted_rew)
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + rew
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError(f"Unknown critic method: {self.critic_method}")

        self._debug_print(f"  [DEBUG] target_values (with entropy) - "
              f"min: {self.target_values.min().item():.4f}, max: {self.target_values.max().item():.4f}, "
              f"mean: {self.target_values.mean().item():.4f}, has_nan: {torch.isnan(self.target_values).any().item()}")

    def get_alpha(self, detach=True, scalar=False):
        """Get entropy coefficient alpha."""
        if self.entropy_coef is not None:
            # Fixed entropy coefficient
            return self.entropy_coef
        else:
            # Learnable entropy coefficient
            alpha = self.log_alpha.exp()
            if detach:
                alpha = alpha.detach()
            if scalar:
                alpha = alpha.item()
            return alpha

    def _update_alpha(self, entropy):
        """Update entropy coefficient (SAC-style)."""
        alpha = self.get_alpha(detach=False, scalar=False)

        # Optionally unscale alpha when computing alpha loss
        # (if we scaled entropy by target_entropy, we may want to unscale alpha)
        if self.unscale_entropy_alpha:
            if self.scale_by_target_entropy:
                alpha = alpha * abs(self.target_entropy)
            # Note: offset_by_target_entropy doesn't need compensation in alpha

        alpha_loss = (alpha * (entropy.detach().mean() - self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

    def _update_learning_rate(self):
        """Update learning rate based on schedule."""
        if self.lr_schedule == 'linear':
            # Linear decay from initial LR to min_lr over max_epochs
            assert self.max_epochs > 0, "max_epochs must be > 0 for linear schedule"

            if self.critic_lrschedule:
                critic_lr = (self.min_lr - self.critic_lr_init) * float(self.epoch / self.max_epochs) + self.critic_lr_init
                for param_group in self.critic_optim.param_groups:
                    param_group['lr'] = critic_lr

            actor_lr = (self.min_lr - self.actor_lr_init) * float(self.epoch / self.max_epochs) + self.actor_lr_init
            for param_group in self.actor_optim.param_groups:
                param_group['lr'] = actor_lr

            self.last_lr = actor_lr

        elif self.lr_schedule == 'constant':
            # Keep learning rate constant at initial value
            lr = self.actor_lr_init
            self.last_lr = lr

        elif self.lr_schedule == 'kl':
            # Adaptive scheduler based on KL divergence
            if self.avg_kl is not None:
                actor_lr = adaptive_scheduler(
                    self.last_lr, self.avg_kl.item(),
                    min_lr=self.min_lr, max_lr=self.max_lr
                )

                if self.critic_lrschedule:
                    critic_lr = actor_lr
                    for param_group in self.critic_optim.param_groups:
                        param_group['lr'] = critic_lr

                for param_group in self.actor_optim.param_groups:
                    param_group['lr'] = actor_lr

                self.last_lr = actor_lr
        else:
            raise NotImplementedError(f"Unknown lr_schedule: {self.lr_schedule}")

    def _log_metrics(self, metrics):
        """Log training metrics (placeholder for tensorboard/wandb)."""
        pass

    def _print_progress(self, metrics):
        """Print training progress."""
        mean_ep_reward = np.mean(self.episode_rewards_hist[-100:]) if self.episode_rewards_hist else 0.0
        mean_ep_length = np.mean(self.episode_lengths_hist[-100:]) if self.episode_lengths_hist else 0.0

        print(f"Epoch {self.epoch} | Steps {self.agent_steps:,} | "
              f"Reward {mean_ep_reward:.2f} | Length {mean_ep_length:.1f} | "
              f"Actor Loss {metrics['actor_loss'][0].item():.4f} | "
              f"Critic Loss {np.mean(metrics['critic_loss']):.4f}")

    def save(self, path):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'agent_steps': self.agent_steps,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict() if not self.no_target_critic else None,
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'obs_rms': self.obs_rms.state_dict() if self.normalize_obs else None,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.epoch = ckpt['epoch']
        self.agent_steps = ckpt['agent_steps']
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if not self.no_target_critic and ckpt['critic_target'] is not None:
            self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optim.load_state_dict(ckpt['actor_optim'])
        self.critic_optim.load_state_dict(ckpt['critic_optim'])
        if self.normalize_obs and ckpt['obs_rms'] is not None:
            self.obs_rms.load_state_dict(ckpt['obs_rms'])
        print(f"Checkpoint loaded from {path}")

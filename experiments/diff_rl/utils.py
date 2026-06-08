"""
Utility functions for SHAC training.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CriticDataset(Dataset):
    """
    Dataset for critic training from collected trajectories.
    """
    def __init__(self, batch_size, obs_buf, target_values, drop_last=False):
        """
        Args:
            batch_size: Size of each batch
            obs_buf: Dictionary of observation tensors (T, B, ...)
            target_values: Target values tensor (T, B)
            drop_last: Whether to drop last incomplete batch
        """
        self.batch_size = batch_size
        self.obs_buf = obs_buf
        self.target_values = target_values
        self.drop_last = drop_last

        # Flatten time and batch dimensions
        T, B = target_values.shape
        self.total_samples = T * B
        self.num_batches = self.total_samples // batch_size
        if not drop_last and self.total_samples % batch_size != 0:
            self.num_batches += 1

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.total_samples)

        # Flatten and slice observations
        T, B = self.target_values.shape
        obs_batch = {}
        for k, v in self.obs_buf.items():
            # Flatten (T, B, ...) -> (T*B, ...)
            flat_obs = v.view(T * B, *v.shape[2:])
            obs_batch[k] = flat_obs[start_idx:end_idx]

        # Flatten and slice target values
        flat_targets = self.target_values.view(T * B)
        target_batch = flat_targets[start_idx:end_idx]

        return obs_batch, target_batch


def soft_update(source, target, alpha):
    """
    Polyak averaging for target network updates.
    target = alpha * source + (1 - alpha) * target

    Args:
        source: Source network
        target: Target network
        alpha: Interpolation coefficient (0 = no update, 1 = copy)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(alpha * source_param.data + (1.0 - alpha) * target_param.data)


def grad_norm(parameters):
    """
    Compute the L2 norm of gradients.

    Args:
        parameters: Iterator of network parameters

    Returns:
        grad_norm: Scalar tensor with gradient norm
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return torch.tensor(total_norm)


def policy_kl(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two Gaussian policies.
    KL(N(mu1, sigma1) || N(mu2, sigma2))

    Args:
        mu1, sigma1: Mean and std of first distribution
        mu2, sigma2: Mean and std of second distribution

    Returns:
        kl: KL divergence for each sample
    """
    kl = torch.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5
    return kl.sum(dim=-1)


def adaptive_scheduler(last_lr, kl, kl_threshold=0.008, lr_threshold=0.008,
                       min_lr=1e-6, max_lr=1e-2):
    """
    Adaptive learning rate scheduling based on KL divergence.

    Args:
        last_lr: Current learning rate
        kl: KL divergence between old and new policy
        kl_threshold: Target KL divergence
        lr_threshold: LR adjustment threshold
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate

    Returns:
        new_lr: Adjusted learning rate
    """
    if kl > kl_threshold:
        new_lr = max(last_lr / 1.5, min_lr)
    elif kl < kl_threshold * 0.5:
        new_lr = min(last_lr * 1.5, max_lr)
    else:
        new_lr = last_lr
    return new_lr


class RunningMeanStd(nn.Module):
    """
    Running mean and std computation for normalization.
    """
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float32))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float32))
        self.register_buffer('count', torch.tensor(1e-4, dtype=torch.float32))

    def update(self, x):
        """Update running statistics with new data."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x):
        """Normalize input using running statistics."""
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def denormalize(self, x):
        """Denormalize input."""
        return x * torch.sqrt(self.var + self.eps) + self.mean


class RewardShaper:
    """
    Reward scaling/shaping wrapper.
    """
    def __init__(self, scale=1.0, shift=0.0):
        self.scale = scale
        self.shift = shift

    def __call__(self, reward):
        return (reward + self.shift) * self.scale

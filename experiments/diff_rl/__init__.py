"""
Differentiable RL package for Genesis simulator.
Implements Short-Horizon Actor-Critic (SHAC).
"""
from .shac import SHACAgent
from .models import GaussianActor, EnsembleCritic, MLP
from .utils import (
    CriticDataset, soft_update, grad_norm, policy_kl,
    adaptive_scheduler, RunningMeanStd, RewardShaper
)

__all__ = [
    'SHACAgent',
    'GaussianActor',
    'EnsembleCritic',
    'MLP',
    'CriticDataset',
    'soft_update',
    'grad_norm',
    'policy_kl',
    'adaptive_scheduler',
    'RunningMeanStd',
    'RewardShaper',
]

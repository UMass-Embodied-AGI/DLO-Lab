"""
Neural network models for SHAC agent.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions import Normal


def weight_init_(m, gain=1.0):
    """Orthogonal weight initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def get_activation(act_type, **act_kwargs):
    """Get activation function by name."""
    if act_type is None or act_type.lower() == 'none':
        return nn.Identity()
    elif act_type.lower() == 'relu':
        return nn.ReLU(**act_kwargs)
    elif act_type.lower() == 'elu':
        return nn.ELU(**act_kwargs)
    elif act_type.lower() == 'tanh':
        return nn.Tanh()
    elif act_type.lower() == 'silu' or act_type.lower() == 'swish':
        return nn.SiLU(**act_kwargs)
    elif act_type.lower() == 'gelu':
        return nn.GELU(**act_kwargs)
    else:
        # Try to get from torch.nn
        try:
            return getattr(nn, act_type)(**act_kwargs)
        except:
            raise ValueError(f"Unknown activation type: {act_type}")


def get_normalization(norm_type, size, **norm_kwargs):
    """Get normalization layer by name."""
    if norm_type is None or norm_type.lower() == 'none':
        return nn.Identity()
    elif norm_type.lower() == 'layernorm':
        return nn.LayerNorm(size, **norm_kwargs)
    elif norm_type.lower() == 'batchnorm':
        return nn.BatchNorm1d(size, **norm_kwargs)
    else:
        # Try to get from torch.nn
        try:
            return getattr(nn, norm_type)(size, **norm_kwargs)
        except:
            raise ValueError(f"Unknown normalization type: {norm_type}")


class TanhTransform(D.Transform):
    """Tanh transform for squashing distributions."""
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(D.TransformedDistribution):
    """Normal distribution squashed by tanh to [-1, 1]."""
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale

        try:
            self.base_dist = D.Normal(loc, scale)
        except (AssertionError, ValueError) as e:
            print(loc)
            print(torch.where(torch.isnan(loc)))
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=validate_args)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        """Compute entropy via sampling."""
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        # return -log_p.mean(axis=0).sum(axis=2)
        return -log_p.mean(axis=0)  # sum done elsewhere


class Dist(nn.Module):
    """Distribution wrapper for actor outputs."""
    def __init__(
        self,
        dist_type='normal',
        minstd=1.0,
        maxstd=1.0,
        minlogstd=None,
        maxlogstd=None,
        validate_args=None,
    ):
        super().__init__()
        self.dist_type = dist_type
        if minlogstd is not None:
            minstd = np.exp(minlogstd)
        if maxlogstd is not None:
            maxstd = np.exp(maxlogstd)
        self.minstd = minstd
        self.maxstd = maxstd
        self.minlogstd = minlogstd
        self.maxlogstd = maxlogstd
        self.validate_args = validate_args

    def forward(self, mu, logstd):
        if self.dist_type == 'normal':
            sigma = torch.exp(logstd)
            distr = Normal(mu, sigma, validate_args=self.validate_args)

        elif self.dist_type == 'squashed_normal':
            if self.minlogstd is not None or self.maxlogstd is not None:
                logstd = torch.clamp(logstd, self.minlogstd, self.maxlogstd)
            sigma = logstd.exp()
            distr = SquashedNormal(mu, sigma, validate_args=self.validate_args)

        else:
            raise NotImplementedError(f"Distribution type '{self.dist_type}' not implemented")

        return mu, sigma, distr

    def __repr__(self):
        if self.dist_type == 'normal':
            std_str = ''
        else:
            if self.minlogstd is not None or self.maxlogstd is not None:
                std_str = f'minlogstd={self.minlogstd}, maxlogstd={self.maxlogstd}'
            else:
                std_str = f'minstd={self.minstd}, maxstd={self.maxstd}'
        return f'Dist(dist_type={self.dist_type}, {std_str})'


class GaussianActor(nn.Module):
    """
    Gaussian policy network that outputs mean and log_std.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256),
                 log_std_bounds=(-5.0, 2.0),
                 activation='relu', norm_type=None,
                 dist_kwargs=None):
        super().__init__()
        self.log_std_bounds = log_std_bounds

        # Build MLP trunk with normalization support
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            # Add normalization if specified
            if norm_type is not None:
                layers.append(get_normalization(norm_type, hidden_dim))

            # Add activation
            layers.append(get_activation(activation))
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(in_dim, action_dim)
        self.log_std_layer = nn.Linear(in_dim, action_dim)

        # Create distribution wrapper
        if dist_kwargs is None:
            dist_kwargs = {'dist_type': 'normal'}
        self.dist = Dist(**dist_kwargs)

        # Initialize weights
        self.apply(lambda m: weight_init_(m, gain=0.01))

        self.float()

    def forward(self, obs, deterministic=False):
        """
        Args:
            obs: (batch, obs_dim) or dict with 'obs' key
            deterministic: If True, return mean without sampling

        Returns:
            mu: (batch, action_dim)
            sigma: (batch, action_dim)
            distr: Distribution object
        """
        # Handle dict input (for compatibility with reference implementation)
        if isinstance(obs, dict):
            obs = obs.get('obs', obs.get('z', obs))

        obs = obs.float()
        h = self.trunk(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])

        # Use Dist wrapper to create distribution
        mu, sigma, distr = self.dist(mu, log_std)

        return mu, sigma, distr


class EnsembleCritic(nn.Module):
    """
    Ensemble of Q-function networks for reduced overestimation bias.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256),
                 num_critics=2, activation='relu', norm_type=None):
        super().__init__()
        self.num_critics = num_critics

        # Create ensemble of critics
        self.critics = nn.ModuleList()
        for _ in range(num_critics):
            layers = []
            in_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))

                # Add normalization if specified
                if norm_type is not None:
                    layers.append(get_normalization(norm_type, hidden_dim))

                # Add activation
                layers.append(get_activation(activation))
                in_dim = hidden_dim

            # Output layer (no activation/norm on final layer)
            layers.append(nn.Linear(in_dim, 1))

            critic = nn.Sequential(*layers)
            self.critics.append(critic)

        # Initialize weights
        self.apply(lambda m: weight_init_(m, gain=1.0))

        self.float()

    def forward(self, obs, return_type='min'):
        """
        Args:
            obs: (batch, obs_dim) or dict with 'obs' key
            return_type: 'min', 'avg', 'all', or 'min_and_avg'

        Returns:
            values: Depending on return_type
        """
        # Handle dict input (for compatibility with reference implementation)
        if isinstance(obs, dict):
            obs = obs.get('obs', obs.get('z', obs))

        obs = obs.float()
        values = [critic(obs) for critic in self.critics]

        if return_type == 'all':
            return values
        elif return_type == 'min':
            return torch.min(torch.stack(values, dim=0), dim=0)[0]
        elif return_type == 'avg':
            return torch.mean(torch.stack(values, dim=0), dim=0)
        elif return_type == 'min_and_avg':
            stacked = torch.stack(values, dim=0)
            min_val = torch.min(stacked, dim=0)[0]
            avg_val = torch.mean(stacked, dim=0)
            return min_val, avg_val
        else:
            raise ValueError(f"Unknown return_type: {return_type}")


class MLP(nn.Module):
    """
    Simple MLP for observation encoding if needed.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256),
                 activation='relu', norm_type=None):
        super().__init__()
        self.out_dim = output_dim

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            # Add normalization if specified
            if norm_type is not None:
                layers.append(get_normalization(norm_type, hidden_dim))

            # Add activation
            layers.append(get_activation(activation))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(lambda m: weight_init_(m, gain=1.0))

        self.float()

    def forward(self, x):
        x = x.float()
        return self.network(x)

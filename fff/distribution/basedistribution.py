"""Base module to provide distributions"""

from abc import ABC, abstractmethod

import torch


class BaseDistribution(ABC):
    """Abstract definition of probability distribution"""

    @abstractmethod
    def mean_neg_log_prob(self, z_xv):
        """Mean negative log probability"""

    @abstractmethod
    def sample(self, n_samples):
        """Sample from distribution"""

    def _remove_mean(self, x):
        mean = torch.mean(x, axis=1, keepdims=True)
        x = x - mean
        return x

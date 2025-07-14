"""Code to define gaussian distribution"""

import numpy as np
import torch

from .basedistribution import BaseDistribution


class PositionVolumePrior(BaseDistribution):
    """Normalizing flow Gaussian latent distribution"""

    def __init__(self, n_dim, n_volume, n_particles):
        self.n_dim = n_dim
        self.n_volume = n_volume
        self.n_particles = n_particles

    def neg_log_prob(self, z_xv, normalizing_constant=False):
        """Negative log probability coordinates and volume"""
        if not z_xv.shape[0] == self.n_particles * self.n_dim + self.n_volume:
            raise ValueError

        z_x = z_xv[: self.n_particles * self.n_dim]
        z_v = z_xv[self.n_particles * self.n_dim :]

        neg_log_pz_x = self.gaussian_neg_log_likelihood(
            z_x, mean_zero=True, normalizing_constant=normalizing_constant
        )

        neg_log_pz_v = self.gaussian_neg_log_likelihood(
            z_v, mean_zero=False, normalizing_constant=normalizing_constant
        )

        neg_log_pz = torch.cat((neg_log_pz_x, neg_log_pz_v))
        return neg_log_pz

    def mean_neg_log_prob(self, z_xv):
        """Average negative log probability"""
        return torch.mean(self.neg_log_prob(z_xv))

    def sample(self, n_samples):
        """Draw coordinate and volume samples from Gaussian distribution"""
        z_x = self.sample_no_com_gaussian(
            size=(n_samples, self.n_particles * self.n_dim)
        )
        z_v = self.sample_gaussian(size=(n_samples, self.n_volume))

        return torch.cat([z_x, z_v], -1)

    def gaussian_neg_log_likelihood(self, x, mean_zero, normalizing_constant=False):
        """Calculate negative log likelihood"""
        # r is invariant to a basis change in the relevant hyperplane
        neg_log_px = 0.5 * x.pow(2)

        if normalizing_constant:
            if mean_zero:
                # The relevant hyperplane is (N-1) * D dimensional.
                degrees_of_freedom = (self.n_particles - 1) * self.n_dim
            else:
                degrees_of_freedom = self.n_particles * self.n_dim
            neg_log_px = 0.5 * degrees_of_freedom * np.log(2 * np.pi)

        return neg_log_px

    def sample_gaussian(self, size):
        """Sample from gaussian"""
        return torch.randn(size)

    def sample_no_com_gaussian(self, size):
        """Draw samples from Gaussian with center of mass at 0"""
        x = self.sample_gaussian(size)
        # This projection only works because Gaussian is rotation invariant around
        # zero and samples are independent!
        x_projected = self._remove_mean(x)
        return x_projected

"""Contains code to define the double well distribution"""

import torch
from .basedistribution import BaseDistribution


class DoubleWellPotential:
    """Potential of the double well"""

    def __init__(self, a, b, c, offs):
        self.a = a
        self.b = b
        self.c = c
        self.offs = offs

    def energy(self, r):
        """Return potential energy"""
        return (
            self.a * (r - self.offs)
            + self.b * torch.pow((r - self.offs), 2)
            + self.c * (r - self.offs) ** 4
        )


class DoubleWellDistribution(BaseDistribution):
    """Define double well distirbution"""

    def __init__(self, n_dim, n_particles, a, b, c, offs):
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.a = a
        self.b = b
        self.c = c
        self.offs = offs
        with torch.inference_mode(False):
            self.indices = torch.triu_indices(n_particles, n_particles, offset=1)

        self.energy = DoubleWellPotential(self.a, self.b, self.c, self.offs).energy

    def _calc_distances(self, x):
        x = x.reshape(self.n_particles, self.n_dim)
        diff = x - x[:, None, :]
        d = torch.sqrt(
            torch.sum(torch.pow(diff[self.indices[0], self.indices[1], :], 2), dim=1)
        )
        return d

    def mean_neg_log_prob(self, x):
        """Calculate the average negative log probability"""
        return torch.sum(self.energy(self._calc_distances(x))) / self.n_particles

    def sample(self, shape, n_nodes):
        """Maybe implement via API to MD sim code, run simulation to create samples"""
        raise NotImplementedError

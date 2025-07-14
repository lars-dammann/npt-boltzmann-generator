"""Inner workings of the free-form flow"""

import abc

import lightning as L
import torch
from torch.func import jacrev, jvp, vjp


class BaseFreeFormFlow(L.LightningModule):
    """Inner workings of free-form-flow inclduing estimation of
    volume change with gradient trick"""

    def __init__(
        self,
        loss_weights,
        lr,
        n_hutchinson_samples,
        n_dim,
        n_volume,
        n_particles,
        latent_distribution,
        real_distribution,
        encoder,
        decoder,
    ) -> None:
        super().__init__()

        self.loss_weights = loss_weights
        self.lr = lr
        self.n_hutchinson_samples = n_hutchinson_samples
        self.n_dim = n_dim
        self.n_volume = n_volume
        self.n_particles = n_particles
        self.latent_distribution = latent_distribution
        self.real_distribution = real_distribution
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, data):
        """Encode data"""
        return self.encoder(data)

    def decode(self, latent_data):
        """Decode data"""
        return self.decoder(latent_data)

    def forward(self, data):
        """Encode and decode"""
        return self.decode(self.encode(data))

    def calc_batch_loss(self, data, surrogate, datatype, agg_fn=torch.mean):
        """Calculate negative log probability and reconstruction loss for batch"""
        nll_loss, reconstruction_loss = torch.vmap(
            self.calc_training_loss, randomness="different"
        )(data, surrogate=surrogate, datatype=datatype)
        return agg_fn(nll_loss), agg_fn(reconstruction_loss)

    def calc_training_loss(self, data, surrogate, datatype):
        """Calculate the loss during training"""
        if datatype == "real":
            transformation = self.encode
            inverse_transformation = self.decode
        elif datatype == "latent":
            transformation = self.decode
            inverse_transformation = self.encode
        else:
            raise ValueError

        nll_loss, _, reconstructed_data = self.calc_nll_loss(
            data, transformation, inverse_transformation, surrogate, datatype
        )
        reconstruction_loss = self.calc_mean_reconstruction_loss(
            data, reconstructed_data
        )
        return nll_loss, reconstruction_loss

    def calc_mean_reconstruction_loss(self, data, reconstructed_data):
        """Calculate mean reconstruction loss"""
        return torch.mean((data - reconstructed_data) ** 2)

    def calc_nll_loss(
        self, data, transformation, inverse_transformation, surrogate, datatype
    ):
        """Calculate negative log probability loss"""
        volume_change = None
        reconstructed_data = None

        if surrogate:
            volume_change, transformed_data, reconstructed_data = (
                self.calc_mean_surrogate_vol_change(
                    data, transformation, inverse_transformation
                )
            )
        elif not surrogate:
            transformed_data = transformation(data)
            volume_change, reconstructed_data = self.calc_mean_vol_change(
                transformed_data, inverse_transformation
            )

        if datatype == "real":
            neg_log_prob = self.calc_latent_mean_neg_log_prob(transformed_data)
        elif datatype == "latent":
            neg_log_prob = self.calc_real_mean_neg_log_prob(transformed_data)
        else:
            raise ValueError

        return neg_log_prob - volume_change, transformed_data, reconstructed_data

    def calc_vol_change(self, transformed_data, inverse_transformation):
        """Calculate volume change of probability flow"""

        def _double_output(argument):
            result = inverse_transformation(argument)
            return result, result

        jac, reconstructed_data = jacrev(_double_output, has_aux=True)(transformed_data)
        log_det = jac.slogdet()[1]
        return log_det, reconstructed_data

    def calc_mean_vol_change(self, transformed_data, inverse_transformation):
        """Calculate mean volume change of probability flow"""
        log_det, reconstructed_data = self.calc_vol_change(
            transformed_data, inverse_transformation
        )
        return log_det / self.n_particles, reconstructed_data

    def _sample_hutchinson_vector(self):
        """
        Sample a random vector v of shape (*x.shape, n_hutchinson_samples)
        with scaled orthonormal columns.

        The reference data is used for shape and dtype.

        :param x: Reference data.
        :return:
        """
        return torch.randn(
            self.n_hutchinson_samples,
            self.n_particles * self.n_dim + self.n_volume,
            device=self.device,
        )

    def calc_surrogate_vol_change(self, data, transformation, inverse_transformation):
        """Calcuate estimate of probability flow volume change"""
        surrogate = 0
        vs = self._sample_hutchinson_vector()

        sampled_surrogate, transformed_data, reconstructed_data = torch.vmap(
            self.calc_gradient_trick
        )(
            vs,
            transformation=transformation,
            inverse_transformation=inverse_transformation,
            data=data,
        )
        transformed_data = transformed_data[0]
        reconstructed_data = reconstructed_data[0]
        surrogate = torch.mean(sampled_surrogate, dim=0)

        # Use one element slice to keep structure of output
        return surrogate, transformed_data, reconstructed_data

    def calc_mean_surrogate_vol_change(
        self, data, transformation, inverse_transformation
    ):
        """Calcuate mean of estimate of probability flow volume change"""
        surrogate, transformed_data, reconstructed_data = (
            self.calc_surrogate_vol_change(data, transformation, inverse_transformation)
        )
        return torch.mean(surrogate), transformed_data, reconstructed_data

    def calc_gradient_trick(
        self, v, transformation=None, inverse_transformation=None, data=None
    ):
        """Gradient trick to estimate volume change"""
        # $ v^T f'(x) $ via backward-mode AD
        transformed_data, grad_func = vjp(transformation, data)
        backward_vj = grad_func(v)[0]

        # $ g'(z) v $ via forward-mode AD
        reconstructed_data, forward_jv = jvp(
            inverse_transformation, tuple([transformed_data]), tuple([v])
        )

        return backward_vj * forward_jv.detach(), transformed_data, reconstructed_data

    def calc_latent_neg_log_prob(self, latent_data):
        """Negative log probability of latent distribution"""
        return self.latent_distribution.neg_log_prob(latent_data)

    def calc_latent_mean_neg_log_prob(self, latent_data):
        """Mean negative log probability of latent distribution"""
        return self.latent_distribution.mean_neg_log_prob(latent_data)

    def calc_real_mean_neg_log_prob(self, data):
        """Mean negative log probability of coordinate distribution"""
        return self.real_distribution.mean_neg_log_prob(data)

"""Contains the code to define the free-form-flow architecture"""

import torch

from base import BaseFreeFormFlow
import distribution
import transformation


class FreeFormFlow(BaseFreeFormFlow):
    """Free-form-flow architecture of the Boltzmann generator"""

    def __init__(
        self,
        flowmodel_kwargs,
        latent_distribution_model,
        latent_distribution_kwargs,
        real_distribution_model,
        real_distribution_kwargs,
        encoder_model,
        encoder_kwargs,
        decoder_model,
        decoder_kwargs,
    ) -> None:
        latent_distribution = getattr(distribution, latent_distribution_model)(
            **latent_distribution_kwargs
        )
        real_distribution = getattr(distribution, real_distribution_model)(
            **real_distribution_kwargs
        )

        encoder = getattr(transformation, encoder_model)(**encoder_kwargs)
        decoder = getattr(transformation, decoder_model)(**decoder_kwargs)

        super().__init__(
            **flowmodel_kwargs,
            latent_distribution=latent_distribution,
            real_distribution=real_distribution,
            encoder=encoder,
            decoder=decoder,
        )

    def calc_subloss(self, data, datatype, mode, surrogate=True):
        """Calculate and log portion of the loss values"""
        prestring = datatype
        if surrogate:
            prestring += "_surrogate"

        nll_loss, reconstruction_loss = self.calc_batch_loss(
            data=data, surrogate=surrogate, datatype=datatype
        )
        self.log(f"{mode}_{prestring}_nll_loss", nll_loss, sync_dist=True)
        self.log(
            f"{mode}_{datatype}_reconstruction_loss",
            reconstruction_loss,
            sync_dist=True,
        )
        weighted_nll_loss = (
            self.loss_weights["nll"] * self.loss_weights[datatype] * nll_loss
        )
        weighted_reconstruction_loss = (
            self.loss_weights["reconstruction"]
            * self.loss_weights[datatype]
            * reconstruction_loss
        )
        self.log(
            f"{mode}_weighted_{prestring}_nll_loss", weighted_nll_loss, sync_dist=True
        )
        self.log(
            f"{mode}_weighted_{datatype}_reconstruction_loss",
            weighted_reconstruction_loss,
            sync_dist=True,
        )
        return data.shape[0], weighted_nll_loss + weighted_reconstruction_loss

    def calc_total_loss(self, batch, mode, surrogate=True):
        """Calculate the total loss"""
        surrogate_string = None
        if surrogate:
            surrogate_string = "surrogate_"
        else:
            surrogate_string = ""

        try:
            real_sample_number, real_loss = self.calc_subloss(
                batch["real"], "real", mode, surrogate=surrogate
            )
            self.log(f"{mode}_real_{surrogate_string}loss", real_loss, sync_dist=True)
        except KeyError:
            real_sample_number = 0
            real_loss = 0
        try:
            latent_sample_number, latent_loss = self.calc_subloss(
                batch["latent"], "latent", mode, surrogate=surrogate
            )
            self.log(
                f"{mode}_latent_{surrogate_string}loss", latent_loss, sync_dist=True
            )
        except KeyError:
            latent_sample_number = 0
            latent_loss = 0

        loss = (real_sample_number * real_loss + latent_sample_number * latent_loss) / (
            real_sample_number + latent_sample_number
        )
        self.log(f"{mode}_{surrogate_string}loss", loss, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Run a training step in Lightning"""
        loss = self.calc_total_loss(batch, "training", surrogate=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Run a validation step in Lightning"""
        self.calc_total_loss(batch, "validation", surrogate=True)
        self.calc_total_loss(batch, "validation", surrogate=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

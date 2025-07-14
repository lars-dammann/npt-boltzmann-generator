"""Code to train the Boltzmann generator"""

import os

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from data.mdsim import MDSimDataModule

from fff import FreeFormFlow

default_config = {
    "encoder_model": "EGNN",
    "encoder_kwargs": {
        "n_dim": 3,
        "n_volume": 0,
        "n_particles": 2,
        "hidden_nf": 64,
        "edge_feat_nf": 1,
        "n_layers": 4,
        "attention": True,
        "init_weights_std": 0.1,
        "init_weights_att_gain": 0.1,
        "norm_constant": 1,
    },
    "decoder_model": "EGNN",
    "decoder_kwargs": {
        "n_dim": 3,
        "n_volume": 0,
        "n_particles": 2,
        "hidden_nf": 64,
        "edge_feat_nf": 1,
        "n_layers": 3,
        "attention": True,
        "init_weights_std": 0.1,
        "init_weights_att_gain": 0.1,
        "norm_constant": 1,
    },
    "latent_distribution": "PositionVolumePrior",
    "latent_distribution_kwargs": {"n_dim": 3, "n_volume": 0, "n_particles": 2},
    "real_distribution": "DoubleWellDistribution",
    "real_distribution_kwargs": {
        "n_dim": 3,
        "n_particles": 2,
        "a": 0.0,
        "b": -4.0,
        "c": 0.9,
        "offs": 4.0,
    },
    "data_kwargs": {
        "n_dim": 3,
        "n_volume": 0,
        "n_particles": 2,
        "train_batch_size": 2048,
        "val_batch_size": 1024,
        "test_batch_size": 1024,
        "data_paths": {
            "molecular": "data/molecular/dw-100000samples-2particles-3dim.npy",
            "latent": "data/latent/gaussian-100000samples-2particles-3dim.npy",
        },
        "training_mode": "molecular",
        "validation_mode": "both",
    },
    "flowmodel_kwargs": {
        "n_dim": 3,
        "n_volume": 0,
        "n_particles": 2,
        "loss_weights": {
            "nll": 1.0,
            "reconstruction": 200.0,
            "latent": 1.0,
            "real": 1.0,
        },
        "n_hutchinson_samples": 1,
        "lr": 0.03,
    },
    "checkpoint_path": "checkpoints",
    "gradient_clip_val": 0.15,
}

def train_func(config):
    """Run to train model"""
    seed_everything(1, workers=True)
    torch.set_float32_matmul_precision("medium")

    try:
        config["n_workers"] = int(os.getenv("SLURM_CPUS_PER_TASK"))
    except TypeError:
        pass

    flowmodel = FreeFormFlow(
        flowmodel_kwargs=config["flowmodel_kwargs"],
        latent_distribution_model=config["latent_distribution"],
        latent_distribution_kwargs=config["latent_distribution_kwargs"],
        real_distribution_model=config["real_distribution"],
        real_distribution_kwargs=config["real_distribution_kwargs"],
        encoder_model=config["encoder_model"],
        encoder_kwargs=config["encoder_kwargs"],
        decoder_model=config["decoder_model"],
        decoder_kwargs=config["decoder_kwargs"],
    )

    moleculardata = MDSimDataModule(**config["data_kwargs"])

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project="dw2-train")

    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        deterministic=True,
        max_epochs=-1,
        gradient_clip_val=config["gradient_clip_val"],
        num_nodes=nnodes,
        log_every_n_steps=10,
    )

    trainer.fit(flowmodel, datamodule=moleculardata)


if __name__ == "__main__":

    train_func(default_config)

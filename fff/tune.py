"""Code for hyperparameter optimization of Boltzmann generator"""

import random
import json
import os
from pathlib import Path

import flatdict
import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from data.mdsim import MDSimDataModule
from fff import FreeFormFlow


def update_config(config, new_config):
    """Convinience function to update the default config"""
    flat = flatdict.FlatDict(config)

    for new_key, new_value in new_config.items():
        update_keys = [key for key in flat.keys() if new_key in key.split(":")]
        if len(update_keys) > 0:
            for k in update_keys:
                flat[k] = new_value
        else:
            flat[new_key] = new_value

    return flat.as_dict()


def train_func(input_config):
    """Run the training of the model"""
    seed_everything(1, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Load and unite configs
    with open(Path(__file__).parent.parent / "configs/train_default.json", "r") as f:
        config = json.load(f)
    config = update_config(config, input_config)

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
    wandb_logger = WandbLogger(project="dw2-tune")
    wandb_logger.experiment.config.update(config)

    nnodes = int(os.getenv("SLURM_NNODES"))
    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[RayTrainReportCallback()],
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        deterministic=True,
        max_epochs=-1,
        gradient_clip_val=config["gradient_clip_val"],
        num_nodes=nnodes,
        log_every_n_steps=5,
        # precision="bf16-true"
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(flowmodel, datamodule=moleculardata)


# Set seed for the search algorithms/schedulers
random.seed(1)
np.random.seed(1)

# The maximum training epochs
num_epochs = 100
# Number of sampls from parameter space
num_samples = 50
grace_period = 5
reduction_factor = 2

search_space_config = {
    "train_batch_size": tune.choice([1024, 2048, 4096, 8192]),
    "n_layers": tune.choice([1, 2, 3, 4]),
    "hidden_nf": tune.choice([5, 10, 15, 20, 25, 30]),
    "attention": tune.choice([True, False]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "init_weights_std": tune.uniform(0.1, 0.2),
    "init_weights_att_gain": tune.loguniform(0.01, 1.0),
    "n_hutchinson_samples": tune.choice([1, 10, 20, 50, 100]),
    "gradient_clip_val": tune.loguniform(0.01, 10.0),
}

scheduler = ASHAScheduler(
    max_t=num_epochs, grace_period=grace_period, reduction_factor=reduction_factor
)

scaling_config = ScalingConfig(
    trainer_resources={"CPU": 1},
    use_gpu=True,
    resources_per_worker={"CPU": 37, "GPU": 1},
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_frequency=0,
        checkpoint_score_attribute="validation_loss",
        checkpoint_score_order="min",
    ),
)

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

tuner = tune.Tuner(
    ray_trainer,
    param_space={"train_loop_config": search_space_config},
    tune_config=tune.TuneConfig(
        metric="validation_loss",
        mode="min",
        num_samples=num_samples,
        scheduler=scheduler,
    ),
)

results = tuner.fit()

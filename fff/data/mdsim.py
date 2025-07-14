"""Contains module to provide training, validation and test data"""

from pathlib import Path

import numpy as np
import torch
import lightning as L

from torch.utils.data import DataLoader, random_split
from lightning.pytorch.utilities import CombinedLoader


class MDSimDataModule(L.LightningDataModule):
    """Lightnining Data Moduls that loads the MD simulation data
    for training, validation and test"""

    def __init__(
        self,
        n_dim: int,
        n_particles: int,
        n_volume: int,
        data_paths: dict = {
            "molecular": "data/molecular/dw-100000samples-2particles-3dim.npy",
            "latent": "data/latent/gaussian-100000samples-2particles-3dim.npy.npy",
        },
        training_mode: str = "both",
        validation_mode: str = "both",
        sample_propotion: float = 0.5,
        train_batch_size: int = 32,
        val_batch_size: int = 10,
        test_batch_size: int = 10,
        n_workers: int = 38,
    ) -> None:

        super().__init__()
        for k in data_paths.keys():
            if k not in ("molecular", "latent"):
                raise ValueError(
                    f"Data path keys can be molecular or latent only! Input was: {k}"
                )

        self.data_paths: dict = data_paths
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.test_batch_size: int = test_batch_size
        self.n_dim: int = n_dim
        self.n_particles: int = n_particles
        self.n_volume: int = n_volume
        self.n_workers: int = n_workers

        self.molecular_train_data = None
        self.molecular_val_data = None
        self.molecular_test_data = None

        self.latent_train_data = None
        self.latent_val_data = None
        self.latent_test_data = None

        if training_mode not in ("molecular", "latent", "both"):
            raise ValueError(
                f"""Training mode can be molecular or latent or both only!
                 Input was: {training_mode}"""
            )
        self.training_mode = training_mode
        if validation_mode not in ("molecular", "latent", "both"):
            raise ValueError(
                f"""Validation mode can be molecular or latent or both only!
                 Input was: {validation_mode}"""
            )
        self.validation_mode = validation_mode
        if (sample_propotion >= 1.0) or (sample_propotion <= 0):
            raise ValueError(
                f"Sample proportion needs to lie between 0 and 1!  Input was: {sample_propotion}"
            )

    def setup(self, stage):
        if "molecular" in [self.training_mode, self.validation_mode] or "both" in [
            self.training_mode,
            self.validation_mode,
        ]:
            (
                self.molecular_train_data,
                self.molecular_val_data,
                self.molecular_test_data,
            ) = self.load_data(self.data_paths["molecular"])
        if "latent" in [self.training_mode, self.validation_mode] or "both" in [
            self.training_mode,
            self.validation_mode,
        ]:
            self.latent_train_data, self.latent_val_data, self.latent_test_data = (
                self.load_data(self.data_paths["latent"])
            )

    def load_data(self, path):
        """Load and split the data from path"""
        # First define system dimensionality and a target energy/distribution
        data_load = np.load(
            Path(__file__).parent.parent.parent / path, allow_pickle=True
        ).astype(np.float32)
        if not data_load.shape[1] == self.n_particles * self.n_dim + self.n_volume:
            raise ValueError(
                "Structure of input data not consistent with particle number, dim and volume"
            )
        # Get the particle positions
        atom_positions = data_load[:, : self.n_particles * self.n_dim].reshape(
            -1, self.n_particles, self.n_dim
        )
        # Should not make a difference since generalized Boltzmann distribution
        # is invariant with respect to translation
        atom_positions = self.remove_mean(atom_positions)

        box_length = data_load[
            :,
            self.n_particles * self.n_dim : self.n_particles * self.n_dim
            + self.n_volume,
        ]
        if self.n_volume > 0:
            # Flatten the data again and connect to volume
            all_data = np.hstack(
                (atom_positions.reshape(-1, self.n_particles * self.n_dim), box_length)
            ).astype(np.float16)
        else:
            all_data = atom_positions.reshape(-1, self.n_particles * self.n_dim)
        all_data = torch.from_numpy(
            all_data,
        )
        return random_split(
            all_data, (0.8, 0.1, 0.1), torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        data_loader_dict = {}
        if self.training_mode in ["molecular", "both"]:
            data_loader_dict["real"] = DataLoader(
                self.molecular_train_data,
                batch_size=self.train_batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=True,
            )
        if self.training_mode in ["latent", "both"]:
            data_loader_dict["latent"] = DataLoader(
                self.latent_train_data,
                batch_size=self.train_batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=True,
            )
        return CombinedLoader(data_loader_dict)

    def val_dataloader(self):
        data_loader_dict = {}
        if self.validation_mode in ["molecular", "both"]:
            data_loader_dict["real"] = DataLoader(
                self.molecular_val_data,
                batch_size=self.val_batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=False,
            )
        if self.validation_mode in ["latent", "both"]:
            data_loader_dict["latent"] = DataLoader(
                self.latent_val_data,
                batch_size=self.val_batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=False,
            )
        return CombinedLoader(data_loader_dict)

    def test_dataloader(self):
        data_loader_dict = {}
        if self.validation_mode in ["molecular", "both"]:
            data_loader_dict["real"] = DataLoader(
                self.molecular_test_data,
                batch_size=self.test_batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=False,
            )
        if self.validation_mode in ["latent", "both"]:
            data_loader_dict["latent"] = DataLoader(
                self.latent_test_data,
                batch_size=self.test_batch_size,
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=False,
            )
        return CombinedLoader(data_loader_dict)

    def remove_mean(self, x):
        """Subtracts center of mass from coordinates"""
        mean = np.mean(x, axis=1, keepdims=True)
        x = x - mean
        return x

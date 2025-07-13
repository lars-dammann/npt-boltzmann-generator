# NpT Boltzmann Generator
Implementation of a Boltzmann generator to sample atomic configurations from an isobaric-isothermal ensemble.

## Architecture
The Boltzman generator consists of a normalizing flow, that is implemented as an Free-form flow. For more information about free-form flows see: [Free-form flows: Make Any Architecture a Normalizing Flow](https://proceedings.mlr.press/v238/draxler24a.html).

The encoder and decoder of the free-form flow consist of E(n) equivariant graph neural networks. For more information for the structure of the encoder and decoder see: [E(n) equivariant graph neural networks](https://proceedings.mlr.press/v139/satorras21a.html).

## Installation
To install the code via pip, change directory into the project folder and run
```
pip install .
```
## Project structure
- configs: Contains the search configs for hyperparameters optimization
- data: Contains the training data
- fff: Free-form flow model
- sampler: Different sampler to create training data
- slurm: sbatch script to run hyperparameter optimization

## Hyperparameter optimization
To find the right hyperparameters for the model, run the tune.sbatch file
```
sbatch tune.sbatch
```

## Dependencies
The project uses PyTorch and PyTorch Lightning for training. The training is tracked by wandb and hyperparameter optimization is performed with Ray Tune.

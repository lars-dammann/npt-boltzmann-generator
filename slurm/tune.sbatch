#!/bin/bash
#SBATCH --partition=normal          # For how long can it run?
#SBATCH --time=08:00:00          # For how long can it run?
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --gres=gpu:full:2
#SBATCH --ntasks=1              # How many cpus per mpi proc
#SBATCH --cpus-per-task=37              # How many cpus per mpi proc
#SBATCH --mail-type=END
#SBATCH --job-name=tune-dw
#SBATCH --output=output/output.%j        # Where to write results
#SBATCH --error=output/error.%j

export CUDA_VISIBLE_DEVICES=0,1    # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

nvidia-smi

source $HOME/.bashrc
micromamba activate npt-fff

cd $HOME/repos/cbg/npt-fff/

output_dir=slurm/output/$SLURM_JOBID
mkdir $output_dir

nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > $output_dir/results-file.csv &

time python -u fff/tune.py >> $output_dir/tune.log
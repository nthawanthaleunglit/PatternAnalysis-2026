#!/bin/bash
# Slurm job for Rangpur: trains the baseline CNN, then the reduced ConvNeXt, on one A100.
# Submit from this folder:  sbatch run_train.sh
#SBATCH --job-name=convnext-adni
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=train_%j.out

source ~/venv-comp3710/bin/activate
python train.py --model cnn --workers 8
python train.py --model convnext --workers 8

#!/bin/bash
# Slurm job for Rangpur: fine-tunes ImageNet-pretrained ConvNeXt-B (88 M parameters) on slices enlarged to 384 x 384,
# then evaluates it alone and in an ensemble with the five ConvNeXt-T runs (same patient split, seed 624).
# Download the weights once on the login node first:
#   python -c "from modules import build_model; build_model('convnext_pt', arch='base')"
# Then submit from this folder:  sbatch run_base384.sh
#SBATCH --job-name=convnext-b384
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=base384_%j.out

source ~/venv-comp3710/bin/activate
python train.py --model convnext_pt --arch base --img-size 384 --split-seed 624 --val-from test \
    --label-smoothing 0.1 --epochs 15 --warmup 1 --lr 5e-5 --drop-path 0.3 --patience 5 --workers 4 \
    --out-dir outputs_base384
python evaluate.py --model convnext_pt --workers 4 --runs outputs_base384
python evaluate.py --model convnext_pt --workers 4 \
    --runs outputs_base384 outputs_ft2 outputs_ft3 outputs_ens/s1 outputs_ens/s2 outputs_ens/s3

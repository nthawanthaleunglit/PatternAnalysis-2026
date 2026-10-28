#!/bin/bash
# Slurm job for Rangpur: fine-tunes ImageNet-pretrained ConvNeXt-T on one A100.
# Download the weights once on the login node first (compute nodes may have no internet):
#   python -c "from modules import build_model; build_model('convnext_pt')"
# Then submit from this folder:  sbatch run_finetune.sh
#SBATCH --job-name=convnext-ft
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=finetune_%j.out

source ~/venv-comp3710/bin/activate
python train.py --model convnext_pt --val-from test --label-smoothing 0.1 --epochs 15 --warmup 1 --lr 5e-5 \
    --drop-path 0.3 --patience 5 --workers 4 --out-dir outputs_ft2
python evaluate.py --model convnext_pt --workers 4 --out-dir outputs_ft2

#!/bin/bash
# Slurm job for Rangpur: the regularised ConvNeXt-T recipe with the two shortcut fixes found by explore.py:
#   --balance-slices  only slice numbers both classes share (78-107 on the course copy), sampled evenly per class
#   --brain-scale     each slice rescaled by its own brain intensity, against scanner-generation differences
# One ablation run with slice balancing only, then three seeds with both fixes and their ensemble.
# Submit from this folder:  sbatch run_debias.sh
#SBATCH --job-name=convnext-debias
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=debias_%j.out

source ~/venv-comp3710/bin/activate
COMMON="--model convnext_pt --split-seed 624 --val-from test --label-smoothing 0.1 --mixup 0.4 --erase 0.25
        --ema 0.999 --layer-decay 0.75 --epochs 20 --warmup 1 --lr 1e-4 --drop-path 0.3 --patience 6 --workers 4"

python train.py $COMMON --seed 624 --balance-slices --out-dir outputs_debias/balance_only
python evaluate.py --model convnext_pt --workers 4 --runs outputs_debias/balance_only

for s in 624 1 2; do
    python train.py $COMMON --seed $s --balance-slices --brain-scale --out-dir outputs_debias/s$s
    python evaluate.py --model convnext_pt --workers 4 --runs outputs_debias/s$s
done
python evaluate.py --model convnext_pt --workers 4 --runs outputs_debias/s624 outputs_debias/s1 outputs_debias/s2
cp ensemble_eval.json outputs_debias/ensemble3_eval.json

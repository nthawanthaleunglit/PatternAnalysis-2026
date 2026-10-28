#!/bin/bash
# Slurm job for Rangpur: ConvNeXt-T fine-tuned with a recipe aimed at memorising individual patients less:
# mixup, random erasing, weight EMA and layer-wise learning-rate decay (plus label smoothing and stochastic depth).
# Three seeds on the same patient split, then each run alone, the three together, and all ten ConvNeXt runs together.
# Submit from this folder:  sbatch run_regularised.sh
#SBATCH --job-name=convnext-reg
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=regularised_%j.out

source ~/venv-comp3710/bin/activate
for s in 624 1 2; do
    python train.py --model convnext_pt --seed $s --split-seed 624 --val-from test --label-smoothing 0.1 \
        --mixup 0.4 --erase 0.25 --ema 0.999 --layer-decay 0.75 --epochs 20 --warmup 1 --lr 1e-4 \
        --drop-path 0.3 --patience 6 --workers 4 --out-dir outputs_reg/s$s
    python evaluate.py --model convnext_pt --workers 4 --runs outputs_reg/s$s
done
REG="outputs_reg/s624 outputs_reg/s1 outputs_reg/s2"
python evaluate.py --model convnext_pt --workers 4 --runs $REG
cp ensemble_eval.json outputs_reg/ensemble3_eval.json
python evaluate.py --model convnext_pt --workers 4 --runs $REG outputs_base384 outputs_ft2 outputs_ft3 \
    outputs_ens/s1 outputs_ens/s2 outputs_ens/s3
cp ensemble_eval.json outputs_reg/ensemble9_eval.json

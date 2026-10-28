#!/bin/bash
# Slurm job for Rangpur: trains three more fine-tuned ConvNeXt-T models that differ only in their random seed,
# then evaluates the five-model ensemble (these three + outputs_ft2 + outputs_ft3) with flip test-time augmentation.
# All members use the same patient split (--split-seed 624), so their predictions can be averaged slice by slice.
# Submit from this folder after run_finetune.sh has produced outputs_ft2 and outputs_ft3:  sbatch run_ensemble.sh
#SBATCH --job-name=convnext-ens
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=ensemble_%j.out

source ~/venv-comp3710/bin/activate
for s in 1 2 3; do
    python train.py --model convnext_pt --seed $s --split-seed 624 --val-from test --label-smoothing 0.1 \
        --epochs 15 --warmup 1 --lr 5e-5 --drop-path 0.3 --patience 5 --workers 4 --out-dir outputs_ens/s$s
done
RUNS="outputs_ft2 outputs_ft3 outputs_ens/s1 outputs_ens/s2 outputs_ens/s3"
python evaluate.py --model convnext_pt --workers 4 --tta --runs outputs_ft2
python evaluate.py --model convnext_pt --workers 4 --runs $RUNS
python evaluate.py --model convnext_pt --workers 4 --tta --runs $RUNS

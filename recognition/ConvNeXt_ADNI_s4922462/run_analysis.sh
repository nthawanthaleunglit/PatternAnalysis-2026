#!/bin/bash
# Slurm job for Rangpur: trains the baseline CNN on the same split as the ConvNeXt runs (validation = half of the
# test folder's patients, seed 624), then runs the dilemma analysis (calibration, reject option, per-class metrics,
# cost) for the CNN and the 9-model ConvNeXt ensemble, and draws example and failure-case figures.
# Submit from this folder after run_regularised.sh:  sbatch run_analysis.sh
#SBATCH --job-name=convnext-analysis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=analysis_%j.out

source ~/venv-comp3710/bin/activate
python train.py --model cnn --split-seed 624 --val-from test --epochs 40 --patience 10 --workers 4 --out-dir outputs_cnn2
python analyse.py --cnn outputs_cnn2 --convnext outputs_reg/s624 outputs_reg/s1 outputs_reg/s2 outputs_base384 \
    outputs_ft2 outputs_ft3 outputs_ens/s1 outputs_ens/s2 outputs_ens/s3
python predict.py

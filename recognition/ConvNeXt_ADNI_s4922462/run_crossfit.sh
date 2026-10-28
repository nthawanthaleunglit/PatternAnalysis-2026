#!/bin/bash
# Slurm job for Rangpur: 3-fold cross-fitting on the test folder. Its patients are split into three folds of whole
# people (repeat scans kept together, seed 624). For fold k the model trains on all of train/ plus fold k+2, stops
# early on fold k+1 and is tested on fold k, so it learns from newer-era AD scans (which train/ lacks) and every
# test-folder patient is scored once by a model that never saw that person. evaluate.py then pools the folds.
# Submit from this folder:  sbatch run_crossfit.sh cnn            (quick check of the idea)
#                           sbatch run_crossfit.sh convnext_pt    (ConvNeXt-T, same recipe as run_ensemble.sh)
#SBATCH --job-name=convnext-cf
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=crossfit_%j.out

MODEL=${1:-cnn}
if [ "$MODEL" = cnn ]; then
    OPTS="--epochs 40 --patience 10"
else
    OPTS="--label-smoothing 0.1 --epochs 15 --warmup 1 --lr 5e-5 --drop-path 0.3 --patience 5"
fi

source ~/venv-comp3710/bin/activate
RUNS=""
for f in 0 1 2; do
    python train.py --model $MODEL --folds 3 --fold $f --split-seed 624 $OPTS --workers 4 \
        --out-dir outputs_cf/$MODEL/f$f
    RUNS="$RUNS outputs_cf/$MODEL/f$f"
done
python evaluate.py --model $MODEL --workers 4 --runs $RUNS --out outputs_cf/$MODEL/crossfit_eval.json

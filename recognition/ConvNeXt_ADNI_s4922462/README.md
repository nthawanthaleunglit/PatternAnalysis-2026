# ConvNeXt for Alzheimer's Disease Classification on ADNI

**Author:** nthawanthaleunglit (s4922462) · COMP3710 Project 2.1 (Classification) · ConvNeXt, Hard difficulty

> **Outcome: the 0.80 test-accuracy target was not met.** The best result on patients the models never saw is
> **0.777 patient-level accuracy** (an ensemble of fine-tuned ConvNeXt models). Every configuration tried, from a
> 0.4 M-parameter CNN to an 88 M-parameter ImageNet-pretrained ConvNeXt-B at 384 px, lands between 0.71 and
> 0.78. This README documents what was tried, why validation accuracy was misleading, and where the errors are.
> My submitted solution for Project 2.1 is the GCN on Facebook Page-Page (branch `gnn-facebook`).

## Problem

Classify 2D T1-weighted brain MRI slices from the ADNI dataset as Alzheimer's disease (AD) or cognitively
normal (NC), and report accuracy per slice and per patient (the mean AD probability over a patient's 20 slices).
The task sheet asks for at least 0.80 test accuracy.

## Dependencies

Python 3.11, PyTorch 2.8, torchvision 0.23, Pillow, NumPy, Matplotlib and scikit-learn (see `requirements.txt`).
Pretrained weights are torchvision's ImageNet-1K ConvNeXt-T and ConvNeXt-B. All training ran on one A100 on
Rangpur; every run is seeded (`--seed`, default 624).

## Data and Splits

The course copy (`/home/groups/comp3710/ADNI/AD_NC`) has `train/` and `test/` folders, each with `NC/` and `AD/`
sub-folders of 256 × 240 greyscale JPEG slices named `<id>_<slice>.jpeg`. Every id has exactly 20 slices.

| Folder | NC patients / slices | AD patients / slices |
|---|---|---|
| train | 556 / 11,120 | 520 / 10,400 |
| test | 227 / 4,540 | 223 / 4,460 |

Two split schemes were used, both strictly patient-wise (asserted in `check_no_leakage`):

1. **`--val-from train`** (first runs): 15 % of the train folder's patients per class as validation, and the
   whole test folder (450 patients) as test.
2. **`--val-from test`** (later runs): the whole train folder for training; the test folder's patients split
   50/50 per class (seed 624) into validation (226 patients) and test (224 patients).

**Why scheme 2 was needed.** Under scheme 1 every model scored 0.91–0.99 on validation but 0.73–0.75 on the
test folder. The pretrained ConvNeXt-T, for example, got 0.994 patient accuracy on validation (one patient wrong)
and 0.742 on test. A check of pixel statistics rules out an intensity shift between folders:

| | NC mean / std | AD mean / std |
|---|---|---|
| train folder | 0.148 / 0.248 | 0.138 / 0.240 |
| test folder | 0.148 / 0.247 | 0.138 / 0.238 |

My explanation is that the id in the file name identifies a **scan, not a person**, so the same person can
appear under several ids within the train folder. Validation patients carved out of it are then not truly
unseen, and the models learn to recognise individuals. Validation drawn from the test folder behaves like the
test set (0.79–0.81 vs 0.73–0.78), so it is the only honest guide for early stopping and the decision cut-off.
The cost is a smaller test set (224 patients, so each accuracy has a standard error of about ±0.03).

Slices are centre-cropped to 224 × 224 (random crop, flip, ±10° rotation, ±5 % scale and brightness/contrast
jitter during training) and normalised with the mean and std of training slices only.

## Method

All models are in `modules.py` and trained by `train.py` (AdamW, linear warm-up then cosine decay, bf16 mixed
precision, early stopping on validation accuracy, best checkpoint kept).

- **Baseline CNN** (0.39 M parameters): four conv–BatchNorm–ReLU–max-pool blocks and a linear head.
- **Reduced ConvNeXt from scratch** (4.8 M): the ConvNeXt layout of Liu et al. [1] with depths 2/2/6/2 and widths
  48–384: 4 × 4 patchify stem, blocks of 7 × 7 depthwise conv → LayerNorm → 4× expand → GELU → project, layer
  scale and stochastic depth, LayerNorm + 2 × 2 stride-2 downsampling between stages.
- **Pretrained ConvNeXt-T / ConvNeXt-B** (28 M / 88 M): torchvision's ImageNet weights with a new 2-class head.
  For 1-channel input the stem's RGB filters are summed into one grey filter, which gives the same output as
  feeding the slice as R = G = B (checked numerically, difference < 1e-6).

Variations tried on top of the pretrained model, all selectable by command-line flags:

- `--label-smoothing 0.1`, stronger stochastic depth (0.3) and a lower learning rate (5e-5), to curb
  overconfidence;
- `--context`: **2.5D input**, the slice below, the slice and the slice above as the three RGB channels;
- `--tta` in `evaluate.py`: average each slice's prediction with its left–right mirror image;
- `--arch base --img-size 384`: ConvNeXt-B with the crop enlarged to 384 × 384;
- ensembles: `evaluate.py --runs A B C ...` averages the slice probabilities of several runs that share one
  patient split;
- a **validation-chosen cut-off**: the threshold on the patient AD probability that maximises validation
  accuracy, applied unchanged to test patients.

## Results

### Scheme 1: validation from the train folder, test = all 450 test-folder patients

| Model | Params | Val acc (slice) | Test acc (slice) | Test acc (patient) | Train time |
|---|---|---|---|---|---|
| Baseline CNN | 0.39 M | 0.927 | **0.746** | **0.749** | 12.8 min |
| Reduced ConvNeXt (scratch) | 4.8 M | 0.908 | 0.728 | 0.742 | 16.1 min |
| ConvNeXt-T pretrained | 28 M | 0.985 | 0.747 | 0.742 | 11.8 min |

### Scheme 2: validation = 226 test-folder patients, test = the other 224

| Model | Val acc (patient) | Test acc (patient, cut-off 0.5) | Test acc (patient, val cut-off) | Test AUROC (patient) |
|---|---|---|---|---|
| ConvNeXt-T, label smoothing (seed 624) | 0.814 | 0.754 | 0.754 | 0.821 |
| ConvNeXt-T + flip TTA | 0.792 | 0.746 | 0.768 | 0.820 |
| ConvNeXt-T 2.5D | 0.810 | 0.728 | 0.754 | 0.839 |
| ConvNeXt-T, seeds 1 / 2 / 3 | – | 0.728 / 0.728 / 0.714 | – | – |
| ConvNeXt-B at 384 px | – | – | 0.768 | **0.854** |
| Ensemble of 5 ConvNeXt-T | 0.805 | 0.723 | **0.777** | 0.837 |
| Ensemble of 5 ConvNeXt-T + flip TTA | 0.779 | 0.723 | 0.768 | 0.837 |
| Ensemble of 5 ConvNeXt-T + ConvNeXt-B 384 | – | – | **0.777** | 0.847 |

Per-run JSON files (`outputs*/convnext_pt_summary.json`, `*_eval.json`, `ensemble_eval*.json`) hold the full
numbers, confusion matrices and accuracy by slice position. Inference costs 0.7–0.9 ms per slice for
ConvNeXt-T on the A100 (2.4 ms with 2.5D input), and fine-tuning ConvNeXt-T takes 5–6 minutes.

### Where the errors are

The errors are almost all one-sided: unseen AD patients are called NC.

| ConvNeXt-T (scheme 1), 450 test patients | Predicted NC | Predicted AD | Recall |
|---|---|---|---|
| True NC (227) | 225 | 2 | 0.99 |
| True AD (223) | 114 | 109 | 0.49 |

- **Not a slice-selection problem.** Test accuracy is flat across the 20 slice positions (0.71–0.77 for every
  position), so restricting the patient vote to "informative" slices would not help.
- **Partly a cut-off problem.** Patient AUROC is 0.82–0.87, so the models do rank AD above NC; the 0.5 cut-off
  is simply in the wrong place for unseen patients. Choosing the cut-off on honest validation moves it as low
  as 0.08 and trades missed AD patients for false alarms (5-model ensemble: AD recall 0.49 → 0.74, NC recall
  0.96 → 0.81), which lifts accuracy from 0.72 to 0.78 but not past 0.80.
- **Capacity is not the bottleneck.** The 0.39 M-parameter CNN and the 88 M-parameter ConvNeXt-B are within
  three points of each other. ConvNeXt-B ranks patients best (AUROC 0.854) but that does not become accuracy.

### Comparison with other students' published results

Public COMP3710 pull requests on the same data report similar numbers: 0.689 patient accuracy with a strict
patient split [2], 0.782 slice accuracy for ConvNeXt-S at 384 px with flip TTA and a tuned threshold [3], and
0.802 patient accuracy for a pretrained ConvNeXt-B [4]. In [4] the test loader is also used as the validation
loader, so the 0.802 is the best of 30 epochs judged on the test set itself. I did not find a result above 0.80
with model selection kept off the test patients.

## Usage

```bash
python dataset.py                                   # split summary and leakage check
sbatch run_train.sh                                 # baseline CNN and reduced ConvNeXt from scratch (scheme 1)
python -c "from modules import build_model; build_model('convnext_pt')"   # download ConvNeXt-T weights once
sbatch run_finetune.sh                              # ConvNeXt-T, 2.5D, scheme 2 (outputs_ft3)
sbatch run_ensemble.sh                              # three more seeds + 5-model ensemble, with and without TTA
sbatch run_base384.sh                               # ConvNeXt-B at 384 px + 6-model ensemble
python evaluate.py --model convnext_pt --runs outputs_ft2            # evaluate one run
```

`outputs_ft2` (single-channel ConvNeXt-T, scheme 2) was produced by `run_finetune.sh` before its last edit,
which added `--context`; remove that flag and set `--out-dir outputs_ft2` to reproduce it.

### Files

- `dataset.py`: file listing, patient-wise splits, leakage asserts, normalisation statistics, augmentation and
  2.5D neighbour lookup.
- `modules.py`: baseline CNN, reduced ConvNeXt, pretrained ConvNeXt-T/S/B wrapper, `build_model`.
- `train.py`: training loop, warm-up + cosine schedule, early stopping, summary JSON and learning curves.
- `evaluate.py`: slice- and patient-level accuracy, AUROC, confusion matrices, accuracy by slice position,
  validation-chosen cut-off, flip TTA and ensembling.
- `run_*.sh`: Slurm scripts for Rangpur.

## Limitations and Next Steps

- The test set under scheme 2 has only 224 patients, so differences under about three points are within noise.
- Using half of the test folder for validation departs from "accuracy on the provided test set"; the scheme 1
  rows are the strict version.
- Not tried: treating each patient's 20 slices as one 3D volume, slice-level attention pooling, or
  cross-validation over the test folder (training on half of it and predicting the other half). The last one
  would test the scan-not-person explanation directly but changes what "test accuracy" means.

## Artificial Intelligence Usage Disclosure

- **Tool:** Claude (Anthropic), through Claude Code.
- **What it helped with:** scaffolding the four Python files and Slurm scripts in small commits; a line-by-line
  walkthrough of the code; diagnosing the validation–test gap (confusion matrices, pixel statistics, accuracy by
  slice position); proposing and implementing the fine-tuning, 2.5D, TTA, ensemble and ConvNeXt-B experiments;
  reviewing other students' public pull requests; and drafting this README from the measured results.
- **How it was verified:**
  - Every script was run end to end on a small synthetic copy of the folder layout before each Rangpur job.
  - Leakage checks are executable: the three splits are asserted to share no patient id.
  - The grey-stem conversion was checked numerically against feeding R = G = B.
  - Every number in this README comes from the JSON files written by `train.py` and `evaluate.py` on Rangpur.

## References

1. Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in
   *Proc. CVPR*, 2022.
2. shakes76/PatternAnalysis-2025, pull request 283, "Add ADNI ConvNeXt classifier (Problem 8)".
3. shakes76/PatternAnalysis-2025, pull request 277, "ConvNeXt (AD vs NC) — ADNI Slice Classification".
4. shakes76/PatternAnalysis-2025, pull request 293, "s4693608 - Alzheimer's Classifier (Task 8)".

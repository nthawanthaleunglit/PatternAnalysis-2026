# ConvNeXt for Alzheimer's Disease Classification on ADNI

**Author:** nthawanthaleunglit (s4922462) · COMP3710 Project 2.1 (Classification) · ConvNeXt, Hard difficulty

> **Outcome: the 0.80 test-accuracy target was not met.** The best patient-level test accuracy is **0.790**
> (an ensemble of nine fine-tuned ConvNeXt models), against **0.705** for the baseline CNN on the same patients.
> Exploring the data shows why every model stalls near 0.78: in the provided `train/` folder, **every AD scan comes
> from an older scan-id range than every NC scan**, so the models learn *what kind of scan it is* as well as the
> disease. Test AD patients from the older range are all caught (23/23); those from the newer range are caught
> about half the time. This README documents the data exploration, the experiments, and the confidence/triage
> investigation the task asks for.

## Problem

Classify T1-weighted brain MRI from ADNI as Alzheimer's disease (AD) or cognitively normal (NC), and decide which
predictions are confident enough to automate and which should be referred to a specialist. The task sheet asks
for at least 0.80 test accuracy, a baseline comparison, metrics beyond accuracy, and a calibrated reject option.

## Dependencies

Python 3.11, PyTorch 2.8, torchvision 0.23, Pillow, NumPy, Matplotlib and scikit-learn (`requirements.txt`).
Pretrained weights are torchvision's ImageNet-1K ConvNeXt-T and ConvNeXt-B. All training ran on one A100 on
Rangpur. Every run is seeded (`--seed`, default 624).

## Data Exploration and Leakage (`explore.py`)

The course copy (`/home/groups/comp3710/ADNI/AD_NC`) has `train/` and `test/` folders with `NC/` and `AD/`
sub-folders of 256 × 240 greyscale **sagittal** slices named `<id>_<slice>.jpeg`. Every id has exactly 20
consecutive slice numbers.

| Folder | NC ids / slices | AD ids / slices |
|---|---|---|
| train | 556 / 11,120 | 520 / 10,400 |
| test | 227 / 4,540 | 223 / 4,460 |

**Checks that passed.** No id is in both classes or both folders; no two of the 30,520 files are byte-identical;
mean pixel intensity is the same in both folders (NC 0.148, AD 0.138 in each).

**Repeat scans under different ids.** `explore.py` shrinks each id's 20 slices, removes the brain shape all scans
share, and compares every pair of ids. Pairs with cosine similarity above 0.9 are almost always the same label
(145 of 146 in `train/`), and the most similar pairs have consecutive ids and look identical (figure below), so
they are the same person scanned twice. The file-name id identifies a **scan, not a person**.

| Pairs with similarity > 0.9 | train–train | test–test | train–test |
|---|---|---|---|
| Count (same label) | 146 (145) | 47 (47) | 13 (11) |

![Most similar pairs of different ids](images/closest_pairs.png)

Consequences: a validation set carved out of `train/` by id shares people with the training set (the first runs
scored 0.99 there but 0.74 on `test/`), and 13 near-identical pairs (11 with the same label) cross between
`train/` and `test/`, so a few people are probably in both folders.

**A shortcut that separates the classes in `train/`.**

| | train AD | train NC | test AD | test NC |
|---|---|---|---|---|
| Scan-id range | 218k – 388k | 809k – 1,442k | 388k – 1,444k (median 985k) | 1,183k – 1,447k |
| Most common first slice of the window | 78, 88, 75 | 78, 94, 88, 95 | 78, 88, 94 | 94, 78, 88 |

ADNI ids grow over time, so all training AD scans come from an earlier acquisition period than all training NC
scans, and the 20-slice window starts at different positions for the two classes (94/95 only for NC, 75 only for
AD). A model can separate the training classes by scan era and slice position without looking at the disease.
Labelling each id by its most similar other id is right 80 % of the time within `train/` but only 57 % from `test/`
to `train/`, which also points to features that do not carry over.

![Nearest-neighbour similarity](images/nn_similarity.png)

**The shortcut explains the test errors.** Test accuracy of the 9-model ConvNeXt ensemble by group (224 patients):

| Test group | Patients | Accuracy | Mean p(AD) |
|---|---|---|---|
| AD, id < 400k (like training AD) | 23 | **1.000** | 0.95 |
| AD, id ≥ 400k (like training NC) | 88 | **0.659** | 0.40 |
| AD, window starting at 94 (NC-only in `train/`) | 23 | **0.391** | 0.12 |
| AD, window starting at 75 (AD-only in `train/`) | 8 | 1.000 | 0.96 |
| NC, window starting at 94 | 52 | 0.981 | 0.05 |

**Fixes tried.** Training only on slice numbers both classes share (78–107), sampled evenly per slice position and
class, and rescaling each slice by its own brain intensity, did *not* help (best 0.768; AD with window 94 fell to
0.04–0.30). The window position travels with the scan era, and the era cannot be removed using `train/` alone
because it contains no newer-era AD scan. A re-split that pools both folders by person would remove it, but it
replaces the provided test set, so it was not used.

## Splits Used

All splits are by id, asserted disjoint in `check_no_leakage`.

1. **Validation from `train/`** (first three runs): 15 % of `train/` ids per class; test = all 450 `test/` ids.
2. **Validation from `test/`** (all later runs): all of `train/` for training; `test/` ids split 50/50 per class
   (seed 624) into validation (226) and test (224). Because of the repeat scans, scheme 1's validation is far too
   optimistic and cannot choose epochs or cut-offs; validation drawn from `test/` behaves like the test set. The
   price is a smaller test set (±0.03 standard error).

Slices are centre-cropped to 224 × 224 (random crop, flip, ±10° rotation, ±5 % scale, brightness/contrast jitter
during training) and normalised with training-slice statistics. A patient's AD probability is the mean over its
slices.

## Models (`modules.py`, `train.py`)

- **Baseline CNN** (0.39 M parameters): four conv–BatchNorm–ReLU–max-pool blocks and a linear head.
- **Reduced ConvNeXt from scratch** (4.8 M): the layout of Liu et al. [1] with depths 2/2/6/2 and widths 48–384:
  4 × 4 patchify stem; blocks of 7 × 7 depthwise conv → LayerNorm → 4× expand → GELU → project, with layer scale and
  stochastic depth; LayerNorm + 2 × 2 stride-2 downsampling between stages.
- **Pretrained ConvNeXt-T / -B** (28 M / 88 M): torchvision's ImageNet weights with a new 2-class head. For one
  input channel the stem's RGB filters are summed, which equals feeding the slice as R = G = B (checked, < 1e-6).

Training: AdamW, linear warm-up then cosine decay, bf16 on GPU, early stopping on validation accuracy. Options
explored (all flags of `train.py`/`evaluate.py`): label smoothing, 2.5D input (neighbouring slices as RGB), flip
test-time augmentation, ConvNeXt-B at 384 px, mixup, random erasing, weight EMA, layer-wise learning-rate decay,
seed ensembles, and a decision cut-off chosen on validation patients.

## Results

### Accuracy across experiments

| Setup | Split | Test patients | Patient accuracy |
|---|---|---|---|
| Baseline CNN | 1 | 450 | 0.749 |
| Reduced ConvNeXt from scratch | 1 | 450 | 0.742 |
| ConvNeXt-T pretrained (val acc 0.994) | 1 | 450 | 0.742 |
| ConvNeXt-T + label smoothing | 2 | 224 | 0.754 |
| ConvNeXt-T 2.5D | 2 | 224 | 0.754 |
| ConvNeXt-B at 384 px | 2 | 224 | 0.768 |
| ConvNeXt-T + mixup, erasing, EMA, layer decay (3 seeds) | 2 | 224 | 0.746 – 0.772 |
| Slice balancing + intensity rescaling (3 seeds) | 2 | 224 | 0.750 – 0.768 |
| Ensemble of 5 ConvNeXt-T | 2 | 224 | 0.777 |
| **Ensemble of 9 (ConvNeXt-T × 8, ConvNeXt-B × 1)** | 2 | 224 | **0.790** |

Split-2 rows use the patient cut-off that maximises validation accuracy. Every configuration, from 0.4 M to
310 M parameters, lands between 0.74 and 0.79. Many configurations were scored on the same 224 test patients, so the best
row is slightly optimistic. Public COMP3710 pull requests on the same data report 0.69 [2], 0.78 [3] and 0.80 [4];
the 0.80 in [4] selects its epoch on the test set.

### Baseline vs ConvNeXt on the same 224 test patients (`analyse.py`)

Both systems are calibrated with Platt scaling, p = sigmoid(a · logit(p_raw) + b), fitted on the 226 validation
patients; with b = 0 this is temperature scaling, and b also moves the decision point, which these models need.

| | Baseline CNN | ConvNeXt ensemble (9) |
|---|---|---|
| Accuracy raw / calibrated | 0.705 / 0.710 | 0.737 / 0.768 |
| Macro-F1 (calibrated) | 0.708 | 0.762 |
| AUROC | 0.778 | **0.843** |
| NC precision / recall / F1 | 0.69 / 0.79 / 0.73 | 0.71 / 0.92 / 0.80 |
| AD precision / recall / F1 | 0.74 / 0.63 / 0.68 | 0.88 / 0.61 / 0.72 |
| ECE raw → calibrated | 0.156 → 0.069 | 0.182 → 0.086 |
| Errors made with ≥ 0.9 confidence (raw) | 25 | **34** |
| Parameters | 0.39 M | 310 M (9 models) |
| Inference, ms per slice (A100, incl. loading) | 0.81 | 9.56 |

![Test accuracy by scan-id era](images/shortcut_by_era.png)

Calibrated accuracy decides at calibrated p = 0.5, which is a smoother rule than the validation-optimal cut-off
behind the 0.790 above; the two differ by 5 of 224 patients.

ConvNeXt's gain is almost entirely on NC patients (recall 0.92 vs 0.79). On newer-era AD patients both models are
at chance (0.53 and 0.51), which confirms the failure is in the data, not the architecture.

## Investigation of the Open Research Dilemma

### 1. Does confidence track correctness?

No, not for raw outputs. Both models are overconfident, ConvNeXt more so: 34 of its wrong patients were called
with at least 90 % confidence (CNN: 25), and its raw calibration error is higher (0.182 vs 0.156). Platt scaling
halves the error for both, but the calibrated probabilities bunch at a few values (226 validation patients are
not many), and the ConvNeXt reliability curve stays jagged because most bins hold only a handful of patients.

![Reliability diagrams](images/reliability.png)

### 2. Reject option

Patients are sorted by calibrated confidence; the most confident are automated and the rest referred.

![Risk-coverage curves](images/risk_coverage.png)

The ConvNeXt ranking is clearly better: its most confident 20 % of test patients are all correct, and about 45 %
can be automated at 90 % accuracy, against about 15 % for the CNN. But the threshold has to be chosen on
validation, and that does not transfer: a threshold set for 90 % on validation gave 85.4 % on test while
automating 70 % of patients, and **21 AD patients were cleared as NC among the automated cases** (CNN: 86.3 % at
33 % coverage, 9 missed AD). The validation and test halves come from the same folder, so the gap is sampling
noise plus the era mix, not a different distribution; it shows how fragile a threshold is with ~200 patients.

### 3. Failure cases (`predict.py`)

![Failure cases](images/failure_cases.png)

- **Confident misses (879209, 1092240; p(AD) 0.15):** newer-era AD scans with the window starting at 78. They look
  like the training NC scans by era, and the model has never seen a newer-era AD brain.
- **Confident false alarms (1265863, 1354011; p(AD) 0.88):** newer-era NC scans, also with the window at 78.
  Nothing in the slices marks them out from correctly classified NC patients, so the model's AD evidence for them
  is not specific; a saliency map would be the next step to see what it responds to.
- **Most uncertain (1334439, p(AD) 0.51):** a newer-era AD scan with the window starting at 94, the hardest group.
- In several patients slice 88 has a visibly different contrast and fold pattern from its neighbours, so the
  20 slices of an id are not always one smooth sweep; this adds noise to the patient average.

Confident correct examples are in `images/examples.png`.

### 4. Recommendation to the project manager

1. **Do not automate clearing of patients as NC.** The worst error in screening is a missed AD patient, and even
   the best model clears newer-protocol AD patients as normal about half the time, often confidently.
2. **Use the model to prioritise, not to decide.** Its ranking is useful (AUROC 0.84; its most confident 20 % are
   all correct): sort the specialist's queue by p(AD) and review every patient, high scores first.
3. **Retrain before any automation.** The training data must contain AD scans from the current acquisition
   protocol; the provided `train/` folder has none. A person-level re-split of all scans would be the first step.
4. **Cost.** The 9-model ensemble costs ~12× the inference time and ~800× the parameters of the CNN for +6 points
   of accuracy and +0.07 AUROC. For prioritisation a single fine-tuned ConvNeXt-T (28 M parameters, ~0.8 ms per
   slice, 0.75–0.77 accuracy) is the better trade-off.

## Usage

```bash
python dataset.py                                   # split summary and leakage asserts
python explore.py                                   # data exploration, repeat scans, shortcut checks
sbatch run_train.sh                                 # baseline CNN and ConvNeXt from scratch (split 1)
python -c "from modules import build_model; build_model('convnext_pt')"   # download weights once (login node)
sbatch run_finetune.sh                              # pretrained ConvNeXt-T, 2.5D (split 2)
sbatch run_ensemble.sh                              # more seeds, ensembles, flip TTA
sbatch run_base384.sh                               # ConvNeXt-B at 384 px
sbatch run_regularised.sh                           # mixup / erasing / EMA / layer decay + 9-model ensemble
sbatch run_debias.sh                                # slice balancing and intensity rescaling
sbatch run_analysis.sh                              # CNN on split 2, dilemma analysis, failure-case figures
python evaluate.py --model convnext_pt --runs outputs_reg/s624     # evaluate one run or an ensemble
```

`outputs_ft2` (single-channel ConvNeXt-T, split 2) came from `run_finetune.sh` before `--context` was added;
drop that flag and use `--out-dir outputs_ft2` to reproduce it.

### Files

- `dataset.py`: file listing, id-wise splits, leakage asserts, normalisation statistics, augmentation, 2.5D
  neighbours, slice balancing and intensity rescaling.
- `modules.py`: baseline CNN, reduced ConvNeXt, pretrained ConvNeXt-T/S/B wrapper, `build_model`.
- `train.py`: training loop, schedules, mixup, erasing, EMA, layer-wise decay, early stopping, summary JSON.
- `evaluate.py`: slice and patient metrics, validation cut-off, flip TTA, ensembles, breakdown by era and window.
- `explore.py`: dataset statistics, duplicate and near-duplicate detection, nearest-neighbour label check.
- `analyse.py`: Platt calibration, ECE, reliability, reject option, per-class metrics, cost.
- `predict.py`: example and failure-case figures.
- `run_*.sh`: Slurm scripts for Rangpur; `images/`: the figures in this README.

## Limitations

- The 224-patient test set gives about ±0.03 standard error, and many configurations were compared on it.
- Validation from `test/` departs from "accuracy on the provided test set"; split 1 is the strict version.
- The repeat-scan grouping and the era explanation are inferred from image similarity and id ranges; ADNI
  metadata (subject id, scan date, scanner) would confirm them but is not part of the course copy.

## Artificial Intelligence Usage Disclosure

- **Tool:** Claude (Anthropic), through Claude Code.
- **What it helped with:** scaffolding the Python files and Slurm scripts in small commits; a line-by-line code
  walkthrough; the data exploration and shortcut analysis; implementing and running the experiments; the
  calibration and reject-option analysis (adapted from my GCN project); reviewing other students' public pull
  requests; and drafting this README from the measured results.
- **How it was verified:**
  - Every script was run end to end on a small synthetic copy of the folder layout before each Rangpur job,
    including fake data with planted duplicates and shifted slice windows to check `explore.py` and slice balancing.
  - Leakage checks are executable: the splits are asserted to share no id.
  - The grey-stem conversion was checked numerically against feeding R = G = B.
  - Every number in this README comes from JSON or console output written on Rangpur.

## References

1. Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in
   *Proc. CVPR*, 2022.
2. shakes76/PatternAnalysis-2025, pull request 283, "Add ADNI ConvNeXt classifier (Problem 8)".
3. shakes76/PatternAnalysis-2025, pull request 277, "ConvNeXt (AD vs NC) — ADNI Slice Classification".
4. shakes76/PatternAnalysis-2025, pull request 293, "s4693608 - Alzheimer's Classifier (Task 8)".
5. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood
   methods," in *Advances in Large Margin Classifiers*, 1999.

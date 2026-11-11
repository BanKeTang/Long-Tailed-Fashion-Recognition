# Long-Tailed Fashion Recognition: controlled experiments

Code and per-seed results for the paper
**"When Long-Tail Remedies Backfire: Rare-Class Augmentation and Re-Sampling in Fine-Tuned Fashion Recognition"**

The study runs controlled, seed-paired experiments on DeepFashion2, DeepFashion1 and CIFAR-100-LT. All configurations use seeds 42, 123 and 2024.

| Study | Question | Scripts | Results |
|---|---|---|---|
| 1 | Does aggressive rare-class augmentation help rare classes, and does the answer depend on model capacity? | `capacity_sweep.py`, `df1_sweep.py`, `cifar_lt_capacity.py` | `results/capacity_*.json`, `results/df1_results.json`, `results/cifar_lt_capacity_results.json` |
| 2 | How do class-balanced loss and re-sampling interact with each other and with rare-class augmentation? (2 × 2 × 2 factorial) | `capacity_sweep.py` (ConvNeXt-Tiny/Large cells), `interaction_runs.py`, `cifar_lt_conflict.py` | `results/phase3_results.json`, `results/phase4_results.json`, `results/interaction_results.json`, `results/reeval_results.json`, `results/cifar_lt_results.json` |
| 3 | How does the cost scale with augmentation magnitude, which component causes it, and is it a feature or a classifier effect? | `capacity_sweep.py` + `rare_policies.py`, `diag_checkpoints.py`, `phase4_diag.py` | `results/phase3_results.json`, `results/diag_results.json`, `results/phase4_diag.json` |
| 4 | Do focal loss and Mixup conflict? | `cifar_lt_conflict.py`, `interaction_runs.py`, `ablation.py` | `results/cifar_lt_results.json`, `results/interaction_results.json`, `results/reeval_results.json` |
| – | Comparison with logit-adjusted loss, classifier re-training (cRT) and class-balanced sampling on the clean baseline | `capacity_sweep.py`, `phase4_diag.py` | `results/phase4_results.json`, `results/phase4_diag.json` |

## Repository layout

```
config.py, dataset.py, model.py, train.py     shared configuration, data pipeline (augmentation policies, 3D-W sampler), classifier, training utilities
rare_policies.py                              parametric rare-class policy (Eq. 2): magnitude levels, geometric-only, photometric-only
prepare_dataset.py                            crop DeepFashion2 garment instances (bounding box + 10% padding)
prepare_df1.py                                crop DeepFashion1 (Category and Attribute Prediction Benchmark)
count_supports.py                             per-class image counts (supports.json)

capacity_sweep.py                             DeepFashion2 runs. Run name = <backbone>_<policy>[_<loss>][_<sampler>]
                                                policy: norare | rare (full) | mild | medium | geom | photo
                                                loss:   ce (CE-LS, default) | cb (CB-LS) | la (logit-adjusted CE)
                                                sampler: 3dw (default) | uni (uniform) | cbs (class-balanced 1/n_c)
df1_sweep.py                                  DeepFashion1 runs (--sampler resample | uniform); val-selected, test-reported
cifar_lt_capacity.py                          CIFAR-100-LT, ResNet-32 width x1..x8 x {off, on}
cifar_lt_conflict.py                          CIFAR-100-LT 2 x 2 designs (CB x re-sampling, focal x Mixup)
interaction_runs.py                           DeepFashion2 ConvNeXt-Large uniform-sampling cells (policy on) and EfficientNet-B3 focal x Mixup
run_combined.py, comparison_methods.py,       earlier-batch runs reused in the paper (see "Earlier-batch runs" below)
repeat_experiments.py, ablation.py

diag_checkpoints.py                           mechanism diagnostics, one checkpoint per configuration (NCM, logit adjustment, weight norms)
phase4_diag.py                                multi-seed diagnostics and classifier re-training (cRT)
reeval_checkpoints.py                         re-evaluate every earlier-batch checkpoint with one evaluation code path
analyze_v3.py                                 paired differences, interactions and statistics for Studies 1-4 (Tables 2-10)
analyze_p4.py                                 ConvNeXt-Large clean cells, DeepFashion1 uniform sampling, other remedies, three-seed diagnostics (Tables 7, 8, 11, 12)
make_figures.py                               reproduce Figures 1-7 (fig1 ... fig7_*.png)
phase2.sh, phase3.sh, phase4.sh,              launchers used for the runs (three parallel queues each)
phase4_resume.sh

results/                                      per-seed results (JSON) used in every table and figure
```

## Reproducing the paper

```bash
pip install -r requirements.txt

# 1. data
python prepare_dataset.py                       # DeepFashion2 -> crops/{train,validation}
python prepare_df1.py --root /path/to/DeepFashion1 --out crops_df1
# CIFAR-100 is downloaded automatically by the CIFAR scripts

# 2. experiments (every script skips finished runs and can be resumed)
python capacity_sweep.py --seed 42 --runs tiny_norare tiny_rare            # Study 1 (repeat for other backbones / seeds)
python capacity_sweep.py --seed 42 --out phase3_results.json \
       --runs tiny_norare_ce_uni tiny_norare_cb_uni tiny_norare_cb \
              tiny_rare_ce_uni tiny_rare_cb_uni tiny_rare_cb \
              tiny_mild tiny_medium tiny_geom tiny_photo                    # Studies 2 and 3
python df1_sweep.py --seeds 42 123 2024
python cifar_lt_conflict.py
python cifar_lt_capacity.py
# or use the launchers: bash phase3.sh / bash phase4.sh

# 3. diagnostics and analysis
python diag_checkpoints.py
python phase4_diag.py
python reeval_checkpoints.py
python analyze_v3.py                            # reads results/*.json
python analyze_p4.py
python make_figures.py
```

`capacity_sweep.py`, `df1_sweep.py`, `interaction_runs.py` and the CIFAR scripts support `--smoke` for a quick end-to-end check.

## Protocol notes

* DeepFashion2 has no public test labels. Checkpoints are selected by overall validation accuracy; for runs with per-epoch histories, the mean of the last five epochs (no selection) is also reported.
* DeepFashion1 uses the official validation split for selection and the official test split for reporting.
* CIFAR-100-LT results are taken after the final epoch.
* Standard deviations are sample standard deviations (n − 1) over three seeds.

## Earlier-batch runs

`reeval_results.json` contains the re-evaluation of checkpoints trained while developing the original pipeline: ConvNeXt-Large with the rare-class policy (CE-LS + 3D-W), CB-LS + 3D-W, CB without label smoothing + uniform sampling, and the EfficientNet-B3 runs of the focal × Mixup study. Checkpoints named `ldam_drw_*` are excluded from the paper: all three were selected within the initial CE-LS + uniform phase of that implementation, before the LDAM loss with deferred re-weighting took over, so they are CE-LS models rather than LDAM-DRW models. Other single-run development checkpoints in that file are listed for completeness and are not used in the paper.

## Data

DeepFashion, DeepFashion2 and CIFAR-100 must be obtained from their original providers under their respective licences. This repository does not redistribute images or model weights.

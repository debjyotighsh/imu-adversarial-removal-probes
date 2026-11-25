# Probe-based analysis of adversarial feature removal in inertial sensing

Code, stored results and figure data for

> Debjyoti Ghosh and Tarun Biswas, *A Probe-Based Analysis of Adversarial Feature Removal in Inertial Sensing for Subject Identification and Activity Recognition*. Submitted to the *Journal of Ambient Intelligence and Smart Environments*, 2026.

The code trains a CNN encoder with gradient-reversal discriminators over a grid of adversarial weights lambda, then fits
post-hoc linear, MLP and k-NN probes to the frozen representation and scores them on segments the encoder never saw.

## What is in this repository

| Path | Content |
|---|---|
| `*.py`, `*.ps1` | training, probing, one run script per experiment, reports and figure scripts (see Layout) |
| `results.zip` | the stored results behind every table and figure: pickled sweeps and CSV files from 87 experiment folders (2.0 MB zipped; no sensor data) |
| `figure_data/` | CSV of every value plotted in Figures 2 to 7 (one row per plotted point: seed mean, sample SD, individual seeds) |
| `paper_figures/` | the figures as printed in the paper (PDF, Times New Roman) and the TikZ source of the overview diagram (Figure 1) |
| `protocol_frozen.txt` | the reporting rule and the four predictions that were fixed before the validation runs |
| `requirements.txt` | the package versions used |

The four datasets are **not** included (see Data below).

## Quick start: regenerate the figures from the stored results

No GPU and no datasets are needed, only numpy, pandas and matplotlib.

```bash
pip install numpy pandas matplotlib
python -m zipfile -e results.zip .                                # creates gait_experiment_output_*/ next to the scripts
python make_times_figures.py --out paper_figures_regenerated      # Figures 2 to 7
python export_figure_data.py --out figure_data_regenerated        # the CSV behind them
```

The regenerated figures match `paper_figures/`, and the regenerated CSV files match `figure_data/`. Times New Roman is used
when it is installed, otherwise STIX (Times-compatible, shipped with matplotlib). Figure 1 is a TikZ diagram:
`cd paper_figures/overview && xelatex overview_times_standalone.tex`. The scripts were developed and run on Windows 11
with Python 3.10; other systems were not tested.

## Install

Python 3.10, one CUDA GPU recommended (CPU works, slowly). Exact versions used are in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data (not redistributed)

The four public datasets must be obtained from their original providers, under their own licences. See the paper's
references for the citations. Check each licence before sharing any derived windows; MobiAct in particular requires
registration with its providers.

| Dataset | Where to get it | Used for |
|---|---|---|
| HHAR | [UCI Machine Learning Repository, dataset 344](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition) (CC BY 4.0) | central condition (device nuisance), most controls |
| RealWorld (HAR) 2016 | [University of Mannheim](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld) | position nuisance, training-set-size and lambda-band results |
| MotionSense | [github.com/mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense) | activity-only conditions, the role-swap experiment |
| MobiAct v2.0 | Biomedical Informatics Laboratory, Hellenic Mediterranean University ([bmi.hmu.gr](https://bmi.hmu.gr/)); available on request under a data-usage agreement | activity-only conditions, 61 subjects |

Build the windowed arrays (50 Hz, 2.56 s windows, 50% overlap, segment ids for leak-free splits):

```bash
python har_datasets.py --dataset realworld --root /path/to/realworld --inspect   # always inspect first
python har_datasets.py --dataset realworld --root /path/to/realworld --build     # likewise motionsense, mobiact
python gait_dual_adversarial_v2.py --hhar_zip /path/to/hhar.zip --activities walk stairsup stairsdown bike  # HHAR
```

**Known gap.** The exact command that produced the cached HHAR file
`windows_hz50_win2.56_ov0.5_bike-stairsdown-stairsup-walk_ex-s3mini_2.npz` was not recorded. Its properties, so a
rebuild can be checked against it, are: devices `nexus4`, `s3`, `s3mini`; activities `bike`, `stairsdown`,
`stairsup`, `walk`; array shape `(32670, 128, 6)`. If your rebuild's shape differs, the exclusion flags
(`--exclude_devices`) are the first thing to check.

Reference shapes and SHA-256 of the cached arrays used in the paper (if a file hash differs after a rebuild,
compare array contents instead, since `.npz` metadata can vary across library versions):

| File | `X` shape | SHA-256 |
|---|---|---|
| HHAR `windows_..._ex-s3mini_2.npz` | (32670, 128, 6) | `3ef783c3b73393ecbbca830237d314e5617dee30896151ecaceb2b0d6b587e4c` |
| RealWorld `acts-jog-stairsdown-stairsup-walk` | (190565, 128, 6) | `b8df429f833bfb11752558a0f9fff27c57324520a73cd91922e81e7ca44966ee` |
| MobiAct `acts-stairsdown-stairsup-walk` | (18587, 128, 6) | `5e68cb0d72bbe04ffcf8dd0f43c20effb11d21edf7cb518985894beed7424840` |
| MobiAct `acts-jog-stand-walk` | (32219, 128, 6) | `df951f4b4ed04f270137d1329723fa2d75fc8762a548a92ef90d1827340ea219` |
| MotionSense `acts-jog-stairsdown-stairsup-walk` | (11601, 128, 6) | `bf252c311078256e714c3bb25fcdb6eb3061946f245e0a30acc034e4b6be7ab3` |
| MotionSense `acts-all` | (21538, 128, 6) | `4d9bd6659012dc7b2aac1695a53bf8a108f406cb8066989e3dec768c63ae2740` |

## Layout

| Path | Role |
|---|---|
| `gait_dual_adversarial_v2.py` | model, training, probes, evaluation, sweep driver (all experiments go through it) |
| `har_datasets.py` | dataset adapters and windowing |
| `seed_stats.py`, `frozen_analysis.py`, `probe_families_table.py` | per-seed removal statistics, confusion-matrix diagnostics, probe-family tables |
| `run_*.ps1` | one script per experiment (PowerShell; each is a plain list of `python` calls you can translate) |
| `*_report.py`, `frozen_*.py`, `make_*figures.py`, `plot_*.py` | read stored results, print table values, draw figures |
| `make_times_figures.py` | draws Figures 2 to 7 as printed in the paper (calls the figure scripts above; Times New Roman) |
| `export_figure_data.py` | writes the CSV of every plotted value (`figure_data/`) |
| `results.zip` | stored results: unzip it here to recreate the `gait_experiment_output_*/` folders the report scripts read |

Results are pickled sweeps in `gait_experiment_output_*/`; every training run is resumable from its checkpoint. The sweeps behind the paper are in `results.zip`. Pickle files can run code when loaded: load only files you trust (the SHA-256 of `results.zip` is below).

## Reproducing each result

Section and table names follow the paper's labels. Run from this directory. To use the stored results instead of retraining, unzip `results.zip` here first (`python -m zipfile -e results.zip .`).

| Result | Run | Then |
|---|---|---|
| Central HHAR sweep, all conditions, three seeds | `run_frozen.ps1` | `frozen_tables.py`, `table_values.py` |
| Five seeds for HHAR central and RealWorld full | `run_extra_seeds.ps1`, `merge_extra_seeds.py` | `merge5_report.py`, `master_numbers.py` |
| Collapse census (204 runs) | uses `run_frozen.ps1` output | `frozen_collapse_stats.py`, `frozen_collapse_removal.py`, `frozen_collapse_detail.py`, `frozen_effect_size.py` |
| Discriminator scored on held-out data | `run_disc_heldout.ps1` | `disc_heldout_table.py` |
| Role swap (activity target, subject nuisance) | `run_subject_nuisance_discheldout.ps1` | `swap_discheldout_report.py`; raw-signal re-identification baseline: `raw_reid_baseline.py` |
| LEACE positive control, permuted-label selectivity control | `run_positive_control.ps1` | `positive_control_report.py` |
| Discriminator restarts (sweep) | `run_disc_restart.ps1` | `disc_restart_report.py` |
| Restart check on MobiAct B and RealWorld (baseline vs restart) | `run_restart_second_dataset.ps1` | `restart_second_report.py`, `restart_second_probes.py`, `restart_second_table.py` (LaTeX rows for the paper table) |
| Slow-encoder / extra-discriminator-step intervention (Table "Slowing the encoder") | `run_intervention.ps1` | `intervention_report.py` |
| Discriminator matching (IIDM, Iwasawa et al. 2020) against gradient reversal | `run_iidm.ps1` | `iidm_report.py` |
| IIDM where there is room to remove (MotionSense subject, HHAR device, RealWorld), faithful configuration, 100-epoch run | `run_iidm_native.ps1`, then `run_iidm_budget.ps1` (200-epoch runs and gradient reversal at the small lambdas; run `-Part A` and `-Part B`) | `iidm_native_report.py`, `iidm_native_summary.py` |
| Factorial test of the mechanism (encoder lr x discriminator updates x restarts, plus lambda=10 controls) | `run_factorial.ps1` (`-Part A` and `-Part B`) | `factorial_report.py` |
| Other encoders (Transformer, BiLSTM) on HHAR and MobiAct B | `run_architectures.ps1` (`-Part T` and `-Part L`) | `arch_report.py` |
| Probe--discriminator gap: probe capacity, fit-set size, encoder training size, epochs, evaluation split (`--probe_grid`) | `run_probe_gap.ps1` (`-Part A`, `-Part B`, `-Part C`) | `probe_gap_report.py`, `probe_gap_figure.py` |
| Recoverability profile: discriminator, three probe families, probe grid, attacker epochs/restarts, segment-bootstrap interval for G, unseen cell (`--probe_grid`) | `run_profile.ps1` (`-Part A`, `-Part B`, `-Part C`) | `profile_report.py` (`--latex` for the table) |
| Protocol validation: B1/B2/P baseline comparison, three new HHAR conditions with the rule frozen first (`protocol_frozen.txt`), and the lambda-selection use case | `run_protocol_validation.ps1` (`-Part A`, `-Part B`, `-Part C`) | `protocol_validation_report.py`, `protocol_rule.py` |
| Held-out subjects: three random sets of whole subjects removed from training and probe fitting (`--hold_subjects_n`, `--hold_subjects_seed`), scored on the `unseen_subject` cell | `run_subject_holdout.ps1` (`-Part A`, `-Part B`, `-Part C`) | `subject_holdout_report.py` (`--latex` for the table) |
| Restart mechanism: per-epoch trace and recovery | `run_restart_dynamics.ps1` | `plot_restart_dynamics.py` |
| Stronger / wider discriminator | `run_disc_stress.ps1` | read `disc_act_acc_raw` and `probe_activity_tc_seen_seen_raw` at lambda=1 |
| Standard discriminator learning rate | in `run_frozen.ps1` (`hhar_dlm1`) | `frozen_dlm1.py` |
| Selection-corrected probe-accuracy reduction (lambda chosen on one half of the evaluation cell, read on the other; five seeds HHAR / RealWorld full, three elsewhere) | `run_selection.ps1` (`-Part A` and `-Part B`) | `selection_report.py`, `protocol_verdicts.py` (verdict table of the protocol) |
| Controlled degradation with lambda chosen on separate data (reruns; the first script skipped some levels, run the fill script after it) | `run_selection_degrade.ps1` (`-Part H` and `-Part M`), `run_selection_degrade_fill.ps1` | `selection_degrade_report.py`, `selection_degrade_figure.py` (the controlled-degradation plot, which is not in the final paper) |
| Controlled degradation | `run_frozen_degrade.ps1` | `frozen_degrade_table.py`, `frozen_degrade_ms.py`, `frozen_fig6.py` (the plot is not in the final paper) |
| Statistical battery | uses `run_frozen.ps1` output | `frozen_stats_battery.py`, `probetype_interaction_model.py`, `stats_significance.py` |
| Figures 2 to 7 as printed in the paper | after the runs above, or from `results.zip` | `make_times_figures.py` (uses `make_seed_figures.py`, `make_probe_figures.py`, `plot_dynamics.py`, `plot_restart_dynamics.py`, `probe_gap_figure.py`); `export_figure_data.py` (CSV of every plotted value) |

**Reconstructed commands.** The training-dynamics run behind the dynamics figure and the TCN replication were launched
interactively and their command lines were not saved. `run_dynamics_and_tcn_RECONSTRUCTED.ps1` rebuilds them from the
paper's description; treat their output as a re-run, not as a verified copy.

## Reproducibility notes

- Seeds fix initialisation and data order, but GPU kernels are not bit-deterministic. Rerunning gives numbers
  that agree within seed noise, not identical digits; the paper's own independent reruns differ by a few thousandths.
- Every figure draws the individual seeds and the sample SD (n-1) across seeds.
- Timing on one consumer GPU: a single-lambda, three-seed HHAR run takes about 2 to 8 minutes; the full HHAR
  eight-lambda sweep about 20 to 40 minutes; RealWorld full pool is the longest.

## Integrity

SHA-256 of `results.zip`: `918c4a6aacebb747bbe9b97836705b89c01b1dc411b4a61b8e9e6ecfc3dde271`

## Licence and citation

The code and the stored results are released under the MIT licence (`LICENSE`). The four datasets keep their own licences.

If you use this code or the results, please cite the paper (see `CITATION.cff`):

> Debjyoti Ghosh and Tarun Biswas. *A Probe-Based Analysis of Adversarial Feature Removal in Inertial Sensing for Subject Identification and Activity Recognition*. Submitted to the *Journal of Ambient Intelligence and Smart Environments*, 2026.

This release has no DOI of its own. To archive it with one, connect the repository to Zenodo and publish a release.

The paper's declaration of generative-AI use also applies to this code.

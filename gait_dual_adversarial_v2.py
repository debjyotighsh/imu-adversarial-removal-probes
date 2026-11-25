"""
Activity-Invariant, Device-Invariant Gait/User Recognition  —  v2
==================================================================
Subject identification from smartphone accelerometer + gyroscope (HHAR),
evaluated for robustness to (a) the activity being performed and (b) the
phone model the signal came from.

WHAT CHANGED FROM v1 (and why)
------------------------------
1. SAMPLING-RATE NORMALIZATION  [critical]
   HHAR sampled each device "at the highest frequency the device allows" —
   that heterogeneity is the dataset's entire reason for existing. v1 windowed
   by SAMPLE COUNT, so a 128-sample window covered a different amount of
   wall-clock time on each phone. "Device" then becomes trivially decodable
   from time-scale alone, and device-adversarial pressure ends up deleting
   CADENCE — which is exactly where gait identity lives. That alone could
   manufacture the "adversarial hurts" result.
   v2 resamples every device onto a common grid (--target_hz) and windows in
   SECONDS (--window_seconds). Run --diagnose first to see the raw rates.

2. LEAK-FREE SPLITS  [correctness]
   v1 shuffled overlapping windows randomly to build the seen/seen cell. With
   50% overlap, a test window shares half its raw samples with a training
   window, so that reference cell was inflated. v2 segments each recording on
   time gaps and assigns WHOLE SEGMENTS to train or test.

3. POOLING THAT KEEPS TEMPORAL STRUCTURE  [capacity]
   v1's AdaptiveAvgPool1d(1) averaged the time axis into 128 numbers, discarding
   the phase/rhythm/asymmetry that carries gait identity — then attached two
   adversarial heads to fight over those 128 dims. v2 offers stats pooling
   (avg|max|std) and attention pooling. --pooling gap reproduces v1.

4. THE LAMBDA SWEEP  [this is the actual contribution]
   Instead of a binary on/off table, sweep the adversarial strength and measure,
   at each point, gait-ID accuracy AGAINST how much activity/device information
   survives in the embedding (linear probe). The resulting Pareto curve is the
   result: identity and activity are not separable factors, and this is the
   measured exchange rate. lambda=0 gives the baseline for free.

5. HONEST METRICS
   Bout-level rank-1 (majority vote over a segment) and balanced accuracy
   alongside raw window accuracy, since the test cells are imbalanced.

USAGE
-----
    # Step 1 — ALWAYS run this first. Prints per-device sampling rates.
    python gait_dual_adversarial_v2.py --hhar_zip /path/to/hhar.zip --diagnose

    # Step 2 — the Pareto sweep (the paper's main figure)
    python gait_dual_adversarial_v2.py --hhar_zip /path/to/hhar.zip \
        --experiment sweep --held_out_device s3mini --held_out_activity walk

    # Optional — the original 4-method table, on corrected data
    python gait_dual_adversarial_v2.py --hhar_zip /path/to/hhar.zip \
        --experiment methods

NOTE ON FRAMING
---------------
HHAR's activities include 'sit' and 'stand', which contain no gait. Training on
static postures and calling the result "gait recognition" will draw a reviewer
objection. Either restrict to ambulatory activities:
    --activities walk stairsup stairsdown bike
and keep the gait framing, or keep all six and call it activity-invariant USER
identification — which is the stronger privacy claim anyway.

Requires: torch, numpy, pandas, scikit-learn, matplotlib
"""

import argparse
import os
import pickle
import warnings
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score

AMBULATORY = ['walk', 'stairsup', 'stairsdown', 'bike']


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Activity/device-invariant gait recognition — v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--hhar_zip', type=str, default=None,
                   help="Path to the HHAR zip, its nested 'Activity recognition exp.zip', "
                        "or a folder already containing the extracted CSVs. Omit when --npz "
                        "is given.")
    p.add_argument('--npz', type=str, default=None,
                   help="A windowed .npz written by har_datasets.py (any dataset). Bypasses "
                        "HHAR loading entirely, so RealWorld / MotionSense run through the "
                        "identical experiment code. Mutually exclusive with --hhar_zip.")
    p.add_argument('--out_dir', type=str, default='./gait_experiment_output_v2')

    p.add_argument('--diagnose', action='store_true',
                   help="Print per-device sampling rates and exit. RUN THIS FIRST.")

    # --- resampling / windowing -------------------------------------------
    p.add_argument('--target_hz', type=float, default=50.0,
                   help="Common resampling rate. 50 Hz is the safe floor across HHAR phones.")
    p.add_argument('--window_seconds', type=float, default=2.56,
                   help="Window length in SECONDS (device-independent).")
    p.add_argument('--overlap', type=float, default=0.5,
                   help="Window overlap fraction. Splits are segment-wise, so this no longer leaks.")
    p.add_argument('--max_gap_s', type=float, default=1.0,
                   help="A time gap longer than this starts a new segment.")
    p.add_argument('--min_segment_s', type=float, default=4.0,
                   help="Discard segments shorter than this.")
    p.add_argument('--time_col', type=str, default='auto',
                   choices=['auto', 'Creation_Time', 'Arrival_Time'],
                   help="'auto' prefers Creation_Time and falls back to Arrival_Time when "
                        "Creation_Time is implausible (a known HHAR quirk on some devices).")

    # --- experimental design ----------------------------------------------
    p.add_argument('--held_out_device', type=str, default='s3mini')
    p.add_argument('--held_out_activity', type=str, default='walk')
    p.add_argument('--activities', type=str, nargs='+', default=None,
                   help="Restrict to these activities. Use the four ambulatory ones to keep "
                        "the 'gait' framing defensible: walk stairsup stairsdown bike")
    p.add_argument('--exclude_devices', type=str, nargs='+', default=None,
                   help="Drop these devices entirely. Matches either the Model column "
                        "('samsungold') or the Device instance column ('s3mini_2'). Use for the "
                        "100 Hz sensitivity run (--exclude_devices samsungold --target_hz 100) "
                        "and to drop truncated instance recordings.")
    p.add_argument('--max_train_windows', type=int, default=0,
                   help="Cap the training set at roughly this many windows by dropping whole "
                        "segments (0 = no cap). Use it to match dataset sizes across corpora: a "
                        "discriminator trained on 15k windows collapses far sooner than one "
                        "trained on 100k, so an apparent difference in 'removability' between "
                        "datasets may be a difference in training-set size. Subsample the larger "
                        "corpus to the smaller one's size before concluding otherwise.")
    p.add_argument('--holdback_frac', type=float, default=0.15,
                   help="Fraction of seen/seen SEGMENTS held out as the in-distribution reference.")

    # --- model / training ---------------------------------------------------
    p.add_argument('--pooling', type=str, default='stats', choices=['gap', 'stats', 'attn'],
                   help="'gap' reproduces v1 (global average — discards temporal structure). "
                        "'stats' concatenates avg|max|std. 'attn' uses learned attention pooling.")
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--disc_lr_mult', type=float, default=1.0,
                   help="Learning-rate multiplier for the activity/device discriminator heads. "
                        "Gradient reversal only exerts meaningful pressure when the discriminator "
                        "is actually strong; if disc_act_acc stays near chance, raise this (5-10) "
                        "before concluding anything from the sweep.")
    p.add_argument('--backbone_lr_mult', type=float, default=1.0,
                   help="Intervention test of the moving-target account: scale the ENCODER's "
                        "learning rate (discriminator rate stays args.lr * disc_lr_mult, i.e. it is "
                        "NOT rescaled with it). 1 = the paper's setting.")
    p.add_argument('--disc_inner_steps', type=int, default=1,
                   help="Intervention test of the moving-target account: discriminator updates per "
                        "encoder update. K>1 adds K-1 discriminator-only steps on the same batch "
                        "(encoder output detached, no gradient reversal) after the joint step. "
                        "1 = the paper's setting.")
    p.add_argument('--hold_subjects_n', type=int, default=0,
                   help="Hold out this many whole SUBJECTS: they are removed from encoder training, the probe fit pool and "
                        "every other cell, and their windows (seen device, seen activity) form the extra cell "
                        "'unseen_subject' on which probes and discriminator heads are scored. Identity is not scored there.")
    p.add_argument('--hold_subjects_seed', type=int, default=0,
                   help="Which random set of subjects --hold_subjects_n holds out (fixed per run, not per training seed).")
    p.add_argument('--probe_grid', action='store_true',
                   help="Also score a grid of probes of different capacity and fit-set size on the seen/seen cell "
                        "(probe-generalisation-gap study). Needs --test_cell_probe.")
    p.add_argument('--split_probe', action='store_true',
                   help="Also score the held-out-cell probes separately on two disjoint halves of the "
                        "seen/seen cell (A for choosing lambda, B for reporting). Needs --test_cell_probe.")
    p.add_argument('--adv_method', choices=['grl', 'iidm'], default='grl',
                   help="How the encoder is pushed against the discriminator. 'grl' = gradient reversal "
                        "(the paper's setting). 'iidm' = invariance induction by discriminator matching "
                        "(Iwasawa, Akuzawa & Matsuo, IJCAI 2020): the discriminator is trained as usual on "
                        "detached features; the encoder minimises, for each sample of nuisance class a, the "
                        "KL between the moving-average discriminator output of every OTHER class i and the "
                        "discriminator's output on that sample.")
    p.add_argument('--iidm_gamma', type=float, default=0.7,
                   help="Decay of the moving-average class centroids in IIDM (0.7 as in the original paper).")
    p.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam',
                   help="Optimiser for backbone and heads. The original IIDM paper used RMSprop.")
    p.add_argument('--constant_lr', action='store_true',
                   help="Constant learning rate instead of the cosine schedule (the original IIDM paper "
                        "used a constant rate).")
    p.add_argument('--iidm_gp', type=float, default=0.0,
                   help="IIDM only: weight of an R1-style gradient penalty on the discriminator "
                        "(squared norm of the gradient of its cross-entropy with respect to its input "
                        "features, Mescheder et al. 2018), as the original paper used. 0 = off.")
    p.add_argument('--iidm_init_centroids', action='store_true',
                   help="IIDM only: initialise the class centroids from the whole training set under the "
                        "initial discriminator, as the original paper did, instead of uniform.")
    p.add_argument('--iidm_constant_lambda', action='store_true',
                   help="IIDM only: use lambda as a constant weight, as in the original paper, instead of the "
                        "DANN-style annealing schedule used everywhere else here.")
    p.add_argument('--disc_restart_every', type=int, default=0,
                   help="Anti-collapse check: reinitialise the discriminator head(s) and their "
                        "optimiser state every N epochs (0 = never, the default). Tests whether "
                        "a standard fix for a discriminator stuck in a degenerate constant "
                        "prediction changes the collapse/recoverability finding.")
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--segment_diagnostics', action='store_true',
                   help="Also record, per seed and per cell, which segment(s) a bout-level "
                        "vote gets right or wrong — the trace needed to explain (rather than "
                        "just report) a coarse cell that lands on a suspiciously exact number "
                        "across seeds/lambdas (e.g. an N-segment cell pinned at k/N every "
                        "time). Adds no training cost, only disk cost. Written to "
                        "'<checkpoint>_segdetail.pkl', keyed by lambda then seed.")

    # --- what to run --------------------------------------------------------
    p.add_argument('--experiment', type=str, default='sweep', choices=['sweep', 'methods'],
                   help="'sweep': lambda sweep + Pareto curve (the main result). "
                        "'methods': the original baseline/activity/device/dual table.")
    p.add_argument('--lambda_sweep', type=float, nargs='+',
                   default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
                   help="Adversarial strengths to sweep. 0.0 == baseline (no gradient reaches "
                        "the encoder) while still measuring natural discriminability.")
    p.add_argument('--mode', type=str, default='adversarial',
                   choices=['adversarial', 'auxiliary'],
                   help="'adversarial': gradient reversal REMOVES activity/device information. "
                        "'auxiliary': plain multi-task, no reversal — the control that separates "
                        "'adversarial removal is harmful' from 'multi-task learning is harmful'.")
    p.add_argument('--lambda_max', type=float, default=1.0,
                   help="Adversarial strength for --experiment methods.")

    p.add_argument('--encoder', type=str, default='cnn', choices=['cnn', 'tcn', 'transformer', 'lstm'],
                   help="Backbone architecture. 'cnn' is the paper's default "
                        "(three plain conv blocks). 'tcn' is the "
                        "architecture-robustness check: a dilated-residual TCN "
                        "with a materially different inductive bias, same "
                        "downstream heads/pooling — use it to check the "
                        "discriminator/probe divergence is not an artifact of "
                        "one encoder choice.")
    p.add_argument('--track_dynamics', action='store_true',
                   help="Record per-epoch training dynamics (discriminator accuracy/"
                        "loss, encoder and head gradient norms, parameter movement, "
                        "representation drift on a fixed batch, post-hoc probe accuracy "
                        "at checkpoints) plus a post-training recovery test (a "
                        "discriminator head trained on the FROZEN encoder). Written to "
                        "'<checkpoint>_dynamics.pkl'. Off by default; training itself is "
                        "unchanged when on. Use a fresh --out_dir: an existing "
                        "checkpoint's lambdas are skipped, so nothing would be recorded.")
    p.add_argument('--disc_hidden', type=int, default=128,
                   help="Hidden width of the discriminator heads (default 128). Raise "
                        "together with --disc_lr_mult to test whether a stronger "
                        "discriminator changes the outcome.")
    p.add_argument('--degrade', type=str, default='none', choices=['none', 'noise', 'mix'],
                   help="Controlled degradation of separability on identical data "
                        "(same subjects, segments, windows, cells): 'noise' adds Gaussian "
                        "noise of --degrade_level x each channel's std to every window; "
                        "'mix' blends each seen-activity window with one from a different "
                        "seen activity with weight --degrade_level (0..0.5). Used to "
                        "manipulate the factor's prior decodability without changing "
                        "anything else about the data.")
    p.add_argument('--degrade_level', type=float, default=0.0,
                   help="Strength of --degrade (noise: multiples of channel std; "
                        "mix: blend weight in [0, 0.5]).")
    p.add_argument('--knn_probe', action='store_true',
                   help="Also fit a k-nearest-neighbour probe (k=15) on every probed "
                        "factor, a second nonlinear family alongside --nonlinear_probe's "
                        "MLP. Together with the linear probe this gives three probe "
                        "families per factor.")
    p.add_argument('--test_cell_probe', action='store_true',
                   help="Also fit each probe family on the TRAINING pool and score it on "
                        "the evaluation cells, whose windows the encoder never saw during "
                        "training ('seen_seen' is the held-back segments of seen devices and "
                        "activities; 'unseen_dev_seen_act' and 'seen_dev_unseen_act' add a "
                        "distribution shift). A factor is scored only on a cell where its "
                        "labels are a subset of the training classes. Stored as "
                        "probe_<factor>[_mlp|_knn]_tc_<cell>. The in-pool probe (segment-"
                        "stratified split of the training pool) is still computed.")
    p.add_argument('--disc_heldout', action='store_true',
                   help="Also score the trained discriminator HEAD ITSELF (not a fresh "
                        "post-hoc probe) on the held-out evaluation cells, the same windows "
                        "the held-out probe uses. Discriminator accuracy is normally reported "
                        "as final-epoch TRAINING accuracy (Sec. 3.6); this adds a held-out "
                        "reading of the same head, so the discriminator and the probe are "
                        "compared on the same data rather than training vs. held-out. Stored "
                        "as disc_<factor>_heldout[_correct|_n|_conf]_tc_<cell>.")
    p.add_argument('--probe_fit_cap', type=int, default=0,
                   help="Cap the number of windows every probe family is fitted on "
                        "(random subsample; 0 = no cap beyond each family's own limit). "
                        "Use the same value for every training-set size so the probes "
                        "cannot differ in fit-set size between conditions.")
    p.add_argument('--nonlinear_probe', action='store_true',
                   help="Also fit a small MLP post-hoc probe (in addition to "
                        "the linear one) on every probed factor. The paper's "
                        "probe is linear by default (Sec. 10.1); this is the "
                        "stricter nonlinear check that section calls for. "
                        "Adds probe-fitting time only, not training time.")

    return p.parse_args()


# =============================================================================
# Extraction (unchanged behaviour from v1)
# =============================================================================
def find_file(root, filename):
    for r, _, files in os.walk(root):
        if '__MACOSX' in r:
            continue
        if filename in files:
            return os.path.join(r, filename)
    return None


def extract_hhar(hhar_zip_path, extract_root):
    marker = os.path.join(extract_root, 'activity_recognition')
    if os.path.exists(marker) and find_file(marker, 'Phones_accelerometer.csv'):
        print("HHAR already extracted, skipping.")
        return marker

    if os.path.isdir(hhar_zip_path):
        print(f"'{hhar_zip_path}' is a folder — searching it directly.")
        if find_file(hhar_zip_path, 'Phones_accelerometer.csv'):
            return hhar_zip_path
        raise FileNotFoundError(
            f"No 'Phones_accelerometer.csv' anywhere under '{hhar_zip_path}'.")

    os.makedirs(extract_root, exist_ok=True)
    print(f"Extracting: {hhar_zip_path}")
    with zipfile.ZipFile(hhar_zip_path, 'r') as z:
        z.extractall(extract_root)

    inner = find_file(extract_root, 'Activity recognition exp.zip')
    if inner is not None:
        print(f"Found nested zip, extracting: {inner}")
        os.makedirs(marker, exist_ok=True)
        with zipfile.ZipFile(inner, 'r') as z:
            z.extractall(marker)

    root = marker if os.path.exists(marker) else extract_root
    if find_file(root, 'Phones_accelerometer.csv'):
        return root
    raise FileNotFoundError("Could not find the phone CSVs after extraction.")


def locate_phone_csvs(extracted_root):
    acc = find_file(extracted_root, 'Phones_accelerometer.csv')
    gyro = find_file(extracted_root, 'Phones_gyroscope.csv')
    assert acc and gyro, "Could not locate Phones_accelerometer.csv / Phones_gyroscope.csv"
    return acc, gyro


# =============================================================================
# Time handling  —  the part that fixes the confound
# =============================================================================
CSV_DTYPES = {'x': np.float32, 'y': np.float32, 'z': np.float32,
              'User': 'category', 'Model': 'category',
              'Device': 'category', 'gt': 'category'}
USECOLS = ['Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt']


def load_phone_csv(path, label):
    print(f"Loading {label} ...")
    df = pd.read_csv(path, usecols=USECOLS, dtype=CSV_DTYPES)
    df = df[df['gt'].notna()]
    df = df[df['gt'].astype(str) != 'null']
    return df.reset_index(drop=True)


def _rate_from(series_ns):
    """Median sampling rate (Hz) implied by a nanosecond timestamp series."""
    if len(series_ns) < 10:
        return np.nan
    dt = np.diff(np.sort(series_ns.to_numpy(dtype=np.float64)))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return np.nan
    return 1e9 / np.median(dt)


def diagnose_rates(df, label):
    """Print the per-device sampling rates implied by both timestamp columns."""
    print(f"\n{'='*78}\nSAMPLING-RATE DIAGNOSTIC — {label}\n{'='*78}")
    print(f"{'Model':<14}{'Device':<14}{'Creation_Time Hz':>18}{'Arrival_Time Hz':>18}{'rows':>12}")
    rows = []
    for (model, dev), g in df.groupby(['Model', 'Device'], observed=True):
        r_create = _rate_from(g['Creation_Time'])
        r_arrive = _rate_from(g['Arrival_Time'] * 1_000_000)  # ms -> ns
        rows.append((str(model), str(dev), r_create, r_arrive, len(g)))
        print(f"{str(model):<14}{str(dev):<14}{r_create:>18.1f}{r_arrive:>18.1f}{len(g):>12,}")

    rates = [r[2] for r in rows if np.isfinite(r[2]) and 1 < r[2] < 2000]
    if rates:
        lo, hi = min(rates), max(rates)
        print(f"\nCreation_Time rate range: {lo:.1f} – {hi:.1f} Hz  (ratio {hi/lo:.2f}x)")
        if hi / lo > 1.2:
            print(
                "\n  >>> Rates differ materially across devices. Windowing by SAMPLE COUNT\n"
                "      would make each window a different duration per phone, letting the\n"
                "      device head key on time-scale and pushing device-adversarial training\n"
                "      to delete cadence. v2 resamples to --target_hz; keep it at or below\n"
                f"      the slowest device ({lo:.0f} Hz).")
        else:
            print("\n  >>> Rates are comparable. The sampling-rate confound is not your problem —\n"
                  "      which makes the negative result more interesting, not less.")
    return rows


def diagnose_coverage(df):
    """
    Per-device-instance coverage. A device instance with far fewer rows, or that
    is missing users/activities its twin has, is a truncated recording — it will
    silently distort any held-out-device cell it lands in. Check before trusting
    a device label that groups it with its twin.
    """
    print(f"\n{'='*78}\nDEVICE COVERAGE\n{'='*78}")
    print(f"{'Model':<14}{'Device':<14}{'users':>8}{'activities':>12}{'rows':>14}")
    for (model, dev), g in df.groupby(['Model', 'Device'], observed=True):
        n_u = g['User'].nunique()
        n_a = g['gt'].nunique()
        print(f"{str(model):<14}{str(dev):<14}{n_u:>8}{n_a:>12}{len(g):>14,}")

    counts = df.groupby('Device', observed=True).size()
    if len(counts) > 1 and counts.min() * 5 < counts.median():
        odd = counts.idxmin()
        print(f"\n  >>> '{odd}' has {counts.min():,} rows against a median of "
              f"{int(counts.median()):,}. That looks truncated. If its user/activity\n"
              f"      coverage is also thinner, drop it: --exclude_devices {odd}")


def choose_time_ns(g, time_col):
    """
    Return a nanosecond timestamp array for one recording group.
    'auto' prefers Creation_Time but falls back to Arrival_Time when Creation_Time
    is implausible — a documented HHAR quirk on some devices.
    """
    if time_col == 'Arrival_Time':
        return g['Arrival_Time'].to_numpy(dtype=np.float64) * 1e6, 'Arrival_Time'
    if time_col == 'Creation_Time':
        return g['Creation_Time'].to_numpy(dtype=np.float64), 'Creation_Time'

    r = _rate_from(g['Creation_Time'])
    if np.isfinite(r) and 5.0 < r < 1000.0:
        return g['Creation_Time'].to_numpy(dtype=np.float64), 'Creation_Time'
    return g['Arrival_Time'].to_numpy(dtype=np.float64) * 1e6, 'Arrival_Time'


def segment_and_resample(t_sec, signal, target_hz, max_gap_s, min_segment_s):
    """
    Split one recording on time gaps, then linearly resample each contiguous
    segment onto a uniform grid at target_hz. Returns a list of (T_i, C) arrays.
    """
    order = np.argsort(t_sec)
    t_sec, signal = t_sec[order], signal[order]

    out = []
    gaps = np.where(np.diff(t_sec) > max_gap_s)[0]
    bounds = np.concatenate([[0], gaps + 1, [len(t_sec)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        ts, sg = t_sec[a:b], signal[a:b]
        if len(ts) < 8:
            continue
        duration = ts[-1] - ts[0]
        if duration < min_segment_s:
            continue
        n_new = int(duration * target_hz) + 1
        grid = ts[0] + np.arange(n_new) / target_hz
        res = np.column_stack([np.interp(grid, ts, sg[:, c]) for c in range(sg.shape[1])])
        out.append(res.astype(np.float32))
    return out


def build_windowed_dataset(acc_path, gyro_path, args, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached windowed data from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X'], d['y_subject'], d['y_activity'], d['y_device'], d['seg_id']

    df_acc = load_phone_csv(acc_path, 'phone accelerometer')
    df_gyro = load_phone_csv(gyro_path, 'phone gyroscope')

    if args.activities:
        keep = set(args.activities)
        df_acc = df_acc[df_acc['gt'].astype(str).isin(keep)]
        df_gyro = df_gyro[df_gyro['gt'].astype(str).isin(keep)]
        print(f"Restricted to activities: {sorted(keep)}")

    if args.exclude_devices:
        drop = set(args.exclude_devices)
        before = len(df_acc)
        m_acc = ~(df_acc['Model'].astype(str).isin(drop) | df_acc['Device'].astype(str).isin(drop))
        m_gy = ~(df_gyro['Model'].astype(str).isin(drop) | df_gyro['Device'].astype(str).isin(drop))
        df_acc, df_gyro = df_acc[m_acc], df_gyro[m_gy]
        print(f"Excluded devices {sorted(drop)}: dropped {before - len(df_acc):,} accelerometer rows")
        assert len(df_acc) > 0, "--exclude_devices removed everything."

    win_len = int(round(args.window_seconds * args.target_hz))
    step = max(1, int(round(win_len * (1.0 - args.overlap))))
    print(f"\nResampling to {args.target_hz:g} Hz; window = {args.window_seconds:g}s "
          f"({win_len} samples), step = {step} samples.")

    windows, y_subject, y_activity, y_device, seg_id = [], [], [], [], []
    seg_counter = 0
    time_sources = {}

    keys = df_acc[['User', 'Device', 'gt']].drop_duplicates().itertuples(index=False)
    for user, dev, activity in keys:
        acc_g = df_acc[(df_acc['User'] == user) & (df_acc['Device'] == dev)
                       & (df_acc['gt'] == activity)]
        gyro_g = df_gyro[(df_gyro['User'] == user) & (df_gyro['Device'] == dev)
                         & (df_gyro['gt'] == activity)]
        if len(acc_g) < 50 or len(gyro_g) < 50:
            continue

        t_acc, src = choose_time_ns(acc_g, args.time_col)
        t_gyro, _ = choose_time_ns(gyro_g, args.time_col)
        time_sources[src] = time_sources.get(src, 0) + 1

        a = pd.DataFrame({'t': t_acc,
                          'ax': acc_g['x'].to_numpy(), 'ay': acc_g['y'].to_numpy(),
                          'az': acc_g['z'].to_numpy()}).sort_values('t')
        gy = pd.DataFrame({'t': t_gyro,
                           'gx': gyro_g['x'].to_numpy(), 'gy': gyro_g['y'].to_numpy(),
                           'gz': gyro_g['z'].to_numpy()}).sort_values('t')

        merged = pd.merge_asof(a, gy, on='t', direction='nearest',
                               tolerance=20_000_000.0)  # 20 ms in ns
        merged = merged.dropna()
        if len(merged) < 50:
            continue

        t_sec = (merged['t'].to_numpy() - merged['t'].to_numpy()[0]) / 1e9
        sig = merged[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].to_numpy(dtype=np.float32)

        model_name = str(acc_g['Model'].iloc[0])
        for seg in segment_and_resample(t_sec, sig, args.target_hz,
                                        args.max_gap_s, args.min_segment_s):
            if len(seg) < win_len:
                continue
            n_before = len(windows)
            for s in range(0, len(seg) - win_len + 1, step):
                windows.append(seg[s:s + win_len])
                y_subject.append(str(user))
                y_activity.append(str(activity))
                y_device.append(model_name)
                seg_id.append(seg_counter)
            if len(windows) > n_before:
                seg_counter += 1

    X = np.asarray(windows, dtype=np.float32)
    y_subject = np.asarray(y_subject)
    y_activity = np.asarray(y_activity)
    y_device = np.asarray(y_device)
    seg_id = np.asarray(seg_id, dtype=np.int64)

    print(f"\nTimestamp source used per group: {time_sources}")
    print(f"X shape: {X.shape}   segments: {seg_counter}")
    print(f"Subjects  ({len(np.unique(y_subject))}): {np.unique(y_subject)}")
    print(f"Activities({len(np.unique(y_activity))}): {np.unique(y_activity)}")
    print(f"Devices   ({len(np.unique(y_device))}): {np.unique(y_device)}")
    if len(np.unique(y_subject)) <= 10:
        print(f"  NOTE: {len(np.unique(y_subject))} subjects means chance is "
              f"{100/len(np.unique(y_subject)):.1f}%. This is the weakest point of any "
              f"identification claim on HHAR — plan to corroborate on OU-ISIR.")

    np.savez_compressed(cache_path, X=X, y_subject=y_subject, y_activity=y_activity,
                        y_device=y_device, seg_id=seg_id)
    print(f"Cached to {cache_path}")
    return X, y_subject, y_activity, y_device, seg_id


# =============================================================================
# Leak-free 2D split  —  whole segments, never individual overlapping windows
# =============================================================================
def build_2d_split(seg_id, y_activity, y_device, held_dev, held_act,
                   holdback_frac=0.15, seed=42):
    dev_seen = y_device != held_dev
    act_seen = y_activity != held_act

    pool_idx = np.where(dev_seen & act_seen)[0]
    segs = np.unique(seg_id[pool_idx])
    rng = np.random.RandomState(seed)
    rng.shuffle(segs)
    # Keep enough segments in the reference cell for bout-level accuracy to mean
    # anything, but never starve training.
    n_test = int(round(holdback_frac * len(segs)))
    n_test = max(min(8, len(segs) // 4), n_test, 1)
    if len(segs) - n_test < 4:
        n_test = max(1, len(segs) // 4)
    test_segs = set(segs[:n_test].tolist())

    is_test = np.isin(seg_id[pool_idx], list(test_segs))
    seen_seen_test_idx = pool_idx[is_test]
    train_idx = pool_idx[~is_test]

    cells = {
        'seen_seen': seen_seen_test_idx,
        'unseen_dev_seen_act': np.where(~dev_seen & act_seen)[0],
        'seen_dev_unseen_act': np.where(dev_seen & ~act_seen)[0],
        'unseen_unseen': np.where(~dev_seen & ~act_seen)[0],
    }
    return train_idx, cells


# =============================================================================
# Model
# =============================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class AttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, h):                      # h: (B, C, T)
        w = torch.softmax(self.score(h), dim=-1)
        return (h * w).sum(dim=-1)             # (B, C)


def _cnn_backbone(in_channels):
    """The paper's default encoder: three plain conv blocks, 32/64/128 channels,
    receptive field grown by max-pooling. Output (B, 128, T')."""
    return nn.Sequential(
        nn.Conv1d(in_channels, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
    )


class _TCNBlock(nn.Module):
    """One dilated-residual TCN block (Bai/Kolter/Koltun style, non-causal since
    windows are offline segments, not a streaming signal). Two dilated convs with
    weight-normalised BN, a 1x1 projection on the residual path when channel
    counts differ, dropout between the two convs."""

    def __init__(self, cin, cout, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.act(h + self.proj(x))


def _tcn_backbone(in_channels):
    """Robustness-check alternative encoder (Sec. 10, encoder-architecture
    check): dilated-residual TCN instead of the default plain CNN. Same
    receptive-field budget as _cnn_backbone (window is 128 samples) reached via
    dilation (1, 2, 4, 8) rather than pooling, so temporal resolution is
    preserved throughout rather than halved twice. Output (B, 128, T),
    T unchanged from input — pooling downstream treats it identically to the
    CNN backbone's (B, 128, T') output."""
    return nn.Sequential(
        _TCNBlock(in_channels, 32, dilation=1),
        _TCNBlock(32, 64, dilation=2),
        _TCNBlock(64, 128, dilation=4),
        _TCNBlock(128, 128, dilation=8),
    )


class _TransformerBackbone(nn.Module):
    """Architecture-generality check: a small Transformer encoder. A strided conv patch embedding
    (kernel 8, stride 4: 128 samples -> 32 tokens, 128 dims), learned positional embedding, two
    pre-norm encoder layers (4 heads, feed-forward 256, dropout 0.1). Output (B, 128, 32)."""

    def __init__(self, in_channels, d=128, layers=2, heads=4, ff=256, n_tok=32):
        super().__init__()
        self.patch = nn.Conv1d(in_channels, d, kernel_size=8, stride=4, padding=2)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff, dropout=0.1,
                                           batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                      # (B, C, T)
        z = self.patch(x).transpose(1, 2)      # (B, tokens, d)
        z = z + self.pos[:, :z.size(1)]
        z = self.norm(self.enc(z))
        return z.transpose(1, 2)               # (B, d, tokens)


class _LSTMBackbone(nn.Module):
    """Architecture-generality check: a two-layer bidirectional LSTM (64 hidden units per direction,
    so 128 features) on the 2x-downsampled signal. Output (B, 128, 64)."""

    def __init__(self, in_channels, hidden=64, layers=2):
        super().__init__()
        self.down = nn.AvgPool1d(2)
        self.rnn = nn.LSTM(in_channels, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=0.1)

    def forward(self, x):
        z = self.down(x).transpose(1, 2)       # (B, T/2, C)
        z, _ = self.rnn(z)                     # (B, T/2, 2*hidden)
        return z.transpose(1, 2)


ENCODERS = {'cnn': _cnn_backbone, 'tcn': _tcn_backbone,
            'transformer': lambda c: _TransformerBackbone(c), 'lstm': lambda c: _LSTMBackbone(c)}


class GaitModel(nn.Module):
    """
    Shared encoder + subject classifier + optional activity/device branches.

    mode='adversarial' : branches sit behind a gradient-reversal layer, so their
                         gradients push the encoder to REMOVE that factor.
    mode='auxiliary'   : no reversal — plain multi-task heads. The control that
                         distinguishes "adversarial removal hurts" from
                         "multi-task learning hurts".
    encoder='cnn'|'tcn' : the backbone architecture. 'cnn' is the paper's
                         default (three plain conv blocks); 'tcn' is the
                         architecture-robustness check (Sec. 10) — a
                         dilated-residual TCN with a materially different
                         inductive bias, same downstream heads/pooling.
    """

    def __init__(self, in_channels=6, num_subjects=9, num_activities=5, num_devices=2,
                 use_activity_branch=True, use_device_branch=True,
                 mode='adversarial', pooling='stats', encoder='cnn', disc_hidden=128):
        super().__init__()
        self.use_activity_branch = use_activity_branch
        self.use_device_branch = use_device_branch
        assert mode in ('adversarial', 'auxiliary')
        self.mode = mode
        self.pooling = pooling

        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder '{encoder}' (have: {sorted(ENCODERS)})")
        self.backbone = ENCODERS[encoder](in_channels)

        if pooling == 'gap':
            self.feat_dim = 128
        elif pooling == 'stats':
            self.feat_dim = 128 * 3
        elif pooling == 'attn':
            self.attn = AttentionPool(128)
            self.feat_dim = 128 * 2            # attention + max
        else:
            raise ValueError(pooling)

        self.identity_classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.feat_dim, 128), nn.ReLU(), nn.Linear(128, num_subjects))

        if use_activity_branch:
            self.grl_act = GradientReversalLayer(1.0)
            self.activity_head = nn.Sequential(
                nn.Linear(self.feat_dim, disc_hidden), nn.ReLU(),
                nn.Linear(disc_hidden, num_activities))
        if use_device_branch:
            self.grl_dev = GradientReversalLayer(1.0)
            self.device_head = nn.Sequential(
                nn.Linear(self.feat_dim, disc_hidden), nn.ReLU(),
                nn.Linear(disc_hidden, num_devices))

    def set_lambdas(self, lam):
        if self.use_activity_branch:
            self.grl_act.lambda_ = lam
        if self.use_device_branch:
            self.grl_dev.lambda_ = lam

    def embed(self, x):
        h = self.backbone(x)                   # (B, 128, T')
        if self.pooling == 'gap':
            return h.mean(dim=-1)
        if self.pooling == 'stats':
            return torch.cat([h.mean(-1), h.max(-1).values, h.std(-1)], dim=1)
        return torch.cat([self.attn(h), h.max(-1).values], dim=1)

    def forward(self, x):
        f = self.embed(x)
        identity_out = self.identity_classifier(f)
        act_out = dev_out = None
        if self.use_activity_branch:
            act_out = self.activity_head(self.grl_act(f) if self.mode == 'adversarial' else f)
        if self.use_device_branch:
            dev_out = self.device_head(self.grl_dev(f) if self.mode == 'adversarial' else f)
        return identity_out, act_out, dev_out


# =============================================================================
# Datasets
# =============================================================================
class GaitDataset(Dataset):
    def __init__(self, X, y_subj, y_act, y_dev):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y_subj = torch.tensor(y_subj, dtype=torch.long)
        self.y_act = torch.tensor(y_act, dtype=torch.long)
        self.y_dev = torch.tensor(y_dev, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y_subj[i], self.y_act[i], self.y_dev[i]


class EvalDataset(Dataset):
    def __init__(self, X, y_subj):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y_subj = torch.tensor(y_subj, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y_subj[i]


# =============================================================================
# Train / evaluate
# =============================================================================
def _iidm_match(head, f, y, Q, gamma):
    """Discriminator-matching loss of IIDM (Iwasawa et al. 2020, eqs. 3-6), averaged over the batch.

    For a sample with nuisance class a the loss is sum over i != a of KL(Q_i || q(.|z)), where Q_i is the
    moving-average discriminator output over class-i representations (updated in place here, from the
    batch, with decay `gamma`). The discriminator's weights are detached, so only the encoder gets this
    gradient; the discriminator itself is trained by its own cross-entropy on detached features.
    Q starts uniform (the original initialises it from all training points; with an untrained head the
    two coincide up to the head's initial bias)."""
    params = {k: v.detach() for k, v in head.named_parameters()}
    logq = torch.log_softmax(torch.func.functional_call(head, params, (f,)), dim=1)
    with torch.no_grad():
        K = Q.size(0)
        oh = torch.nn.functional.one_hot(y, K).float()
        cnt = oh.sum(0)
        present = cnt > 0
        batch_c = (oh.t() @ logq.exp()) / cnt.clamp(min=1).unsqueeze(1)
        Q[present] = gamma * Q[present] + (1.0 - gamma) * batch_c[present]
        mix = (1.0 - oh) @ Q                      # per sample: sum over i != a of Q_i
    return -(mix * logq).sum(1).mean()            # equals the summed KL up to a constant in the encoder


def _r1_penalty(logits, f_in, y):
    """R1-style gradient penalty on a discriminator (Mescheder et al. 2018): mean over the batch of the
    squared norm of the gradient of the discriminator's per-sample cross-entropy with respect to its input
    features. Differentiable, so it trains the discriminator head (the input is detached from the encoder)."""
    ce_sum = torch.nn.functional.cross_entropy(logits, y, reduction='sum')
    g, = torch.autograd.grad(ce_sum, f_in, create_graph=True)
    return g.pow(2).sum(1).mean()


def _grad_norm_of(scalar, params):
    """L2 norm of d(scalar)/d(params), without touching .grad (retains the graph)."""
    gs = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return float(sum(float((g.detach() ** 2).sum()) for g in gs if g is not None) ** 0.5)


def _grad_norm_stored(params):
    """L2 norm of the .grad already accumulated on params by backward()."""
    return float(sum(float((p.grad.detach() ** 2).sum())
                     for p in params if p.grad is not None) ** 0.5)


def _recovery_curves(model, Xtr_n, ysubj_tr, yact_tr, seg_tr, args, dev_torch, seed,
                     n_extra=10):
    """
    Post-training recovery test for the ACTIVITY discriminator, encoder FROZEN.

    If the discriminator read chance during training because the information was
    destroyed, no head can recover it once the encoder stops moving. If it read
    chance because the encoder moved faster than the head could track, a head
    trained on the now-fixed representation should recover quickly. Two heads:
      'cont'  the trained discriminator head, trained further (same capacity)
      'fresh' a new, wider head (256 hidden units) trained from scratch
    Each curve reports accuracy at extra-epoch 0 (before any further training)
    and after every extra epoch, on the probe's fit split (train_acc) and on its
    held-out split (test_acc, test_bal_acc), split by segment exactly as the
    in-pool probe is. This is a held-back split of the TRAINING pool, not the
    'seen_seen' cell used for the discriminator and probe elsewhere in the
    sweep; the two are not directly comparable without noting that.
    """
    import copy
    _, emb = predict(model, Xtr_n, ysubj_tr, args.batch_size, dev_torch)
    fm = segment_stratified_split(seg_tr, yact_tr)
    E = torch.tensor(emb, dtype=torch.float32, device=dev_torch)
    Y = torch.tensor(yact_tr, dtype=torch.long, device=dev_torch)
    fit = torch.tensor(np.where(fm)[0], device=dev_torch)
    tst = torch.tensor(np.where(~fm)[0], device=dev_torch)
    mu, sd = E[fit].mean(0), E[fit].std(0) + 1e-8
    ce = nn.CrossEntropyLoss()

    def curve(head, fwd):
        opt = optim.Adam(head.parameters(), lr=args.lr * args.disc_lr_mult)
        gen = torch.Generator().manual_seed(seed)

        def accs(ep):
            with torch.no_grad():
                a = float((fwd(E[fit]).argmax(1) == Y[fit]).float().mean())
                if len(tst):
                    pred_tst = fwd(E[tst]).argmax(1)
                    b = float((pred_tst == Y[tst]).float().mean())
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        b_bal = float(balanced_accuracy_score(
                            Y[tst].cpu().numpy(), pred_tst.cpu().numpy()))
                else:
                    b = b_bal = float('nan')
            return dict(epoch=ep, train_acc=a, test_acc=b, test_bal_acc=b_bal)

        out = [accs(0)]
        for ep in range(1, n_extra + 1):
            perm = fit[torch.randperm(len(fit), generator=gen).to(dev_torch)]
            for a0 in range(0, len(perm), args.batch_size):
                idx = perm[a0:a0 + args.batch_size]
                opt.zero_grad()
                ce(fwd(E[idx]), Y[idx]).backward()
                opt.step()
            out.append(accs(ep))
        return out

    n_cls = int(yact_tr.max()) + 1
    cont_head = copy.deepcopy(model.activity_head)
    for prm in cont_head.parameters():
        prm.requires_grad_(True)
    torch.manual_seed(seed)
    fresh_head = nn.Sequential(nn.Linear(E.shape[1], 256), nn.ReLU(),
                               nn.Linear(256, n_cls)).to(dev_torch)
    return {'cont': curve(cont_head, lambda z: cont_head(z)),
            'fresh': curve(fresh_head, lambda z: fresh_head((z - mu) / sd))}


def train_model(X, y_subj, y_act, y_dev, train_idx, n_subj, n_act, n_dev,
                use_act, use_dev, args, seed, lam_max, mode, dev_torch, seg_id=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = X[train_idx]
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(axis=0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-8

    ds = GaitDataset((Xtr - mean) / std, y_subj[train_idx], y_act[train_idx], y_dev[train_idx])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = GaitModel(in_channels=X.shape[-1], num_subjects=n_subj,
                      num_activities=n_act, num_devices=n_dev,
                      use_activity_branch=use_act, use_device_branch=use_dev,
                      mode=mode, pooling=args.pooling,
                      encoder=args.encoder,
                      disc_hidden=getattr(args, 'disc_hidden', 128)).to(dev_torch)

    # Give the discriminator heads their own learning rate. Gradient reversal only
    # applies real pressure when the discriminator is strong; a weak one turns the
    # reversed gradient into noise that degrades the encoder without removing
    # anything — which looks exactly like an "invariance/accuracy trade-off".
    head_params, backbone_params = [], []
    for name, prm in model.named_parameters():
        (head_params if ('activity_head' in name or 'device_head' in name)
         else backbone_params).append(prm)
    bb_lr = args.lr * float(getattr(args, 'backbone_lr_mult', 1.0))

    def make_opt():
        groups = [{'params': backbone_params, 'lr': bb_lr},
                  {'params': head_params, 'lr': args.lr * args.disc_lr_mult}]
        if getattr(args, 'optimizer', 'adam') == 'rmsprop':
            o = optim.RMSprop(groups)
        else:
            o = optim.Adam(groups)
        if getattr(args, 'constant_lr', False):
            s = optim.lr_scheduler.LambdaLR(o, lambda e: 1.0)
        else:
            s = optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs)
        return o, s

    opt, sched = make_opt()
    ce = nn.CrossEntropyLoss()
    inner_steps = int(getattr(args, 'disc_inner_steps', 1) or 1)
    iidm = (getattr(args, 'adv_method', 'grl') == 'iidm') and mode == 'adversarial'
    if iidm:
        assert not getattr(args, 'track_dynamics', False), "--track_dynamics is not implemented for iidm"
        Q_act = torch.full((n_act, n_act), 1.0 / n_act, device=dev_torch) if use_act else None
        Q_dev = torch.full((n_dev, n_dev), 1.0 / n_dev, device=dev_torch) if use_dev else None
        iidm_gamma = float(args.iidm_gamma)
        iidm_gp = float(getattr(args, 'iidm_gp', 0.0))
        if getattr(args, 'iidm_init_centroids', False):
            # class centroids Q_i = mean over class-i training windows of the initial discriminator's
            # softmax output, as in the original paper (their "initialized by using all training data")
            model.eval()
            with torch.no_grad():
                for Q, head, lab_all in ((Q_act, getattr(model, 'activity_head', None), ds.y_act),
                                         (Q_dev, getattr(model, 'device_head', None), ds.y_dev)):
                    if Q is None or head is None:
                        continue
                    tot = torch.zeros_like(Q)
                    cnt = torch.zeros(Q.size(0), device=dev_torch)
                    for i in range(0, len(ds), 4096):
                        xb_ = ds.X[i:i + 4096].to(dev_torch)
                        y_ = lab_all[i:i + 4096].to(dev_torch)
                        p_ = torch.softmax(head(model.embed(xb_)), dim=1)
                        tot.index_add_(0, y_, p_)
                        cnt.index_add_(0, y_, torch.ones_like(y_, dtype=torch.float32))
                    ok = cnt > 0
                    Q[ok] = tot[ok] / cnt[ok].unsqueeze(1)
            model.train()

    diag = {'disc_act_acc': float('nan'), 'disc_dev_acc': float('nan')}
    conf_act = np.zeros((n_act, n_act), dtype=np.int64)
    conf_dev = np.zeros((n_dev, n_dev), dtype=np.int64)

    # ---- optional training-dynamics tracking (--track_dynamics) ----------------
    # Adds measurement only: the extra autograd.grad calls reuse the forward pass
    # already done for the update, and the eval-mode passes draw no random numbers,
    # so the training trajectory is the same with tracking on or off.
    track = bool(getattr(args, 'track_dynamics', False)) and seg_id is not None
    if track:
        dyn = []
        Xtr_n = (Xtr - mean) / std
        seg_tr = seg_id[train_idx]
        yact_tr, ydev_tr = y_act[train_idx], y_dev[train_idx]
        gen_fix = torch.Generator().manual_seed(1234)
        fix_idx = torch.randperm(len(ds), generator=gen_fix)[:min(1024, len(ds))]
        X_fix = ds.X[fix_idx].to(dev_torch)          # fixed windows for representation drift
        prev_emb = None
        theta_prev = [p_.detach().clone() for p_ in backbone_params]
        head_prev = [p_.detach().clone() for p_ in head_params]
        probe_epochs = sorted({e for e in (1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 25)
                               if e <= args.epochs} | {args.epochs})
        probe_masks = {k: segment_stratified_split(seg_tr, lab)
                       for k, lab in (('act', yact_tr), ('dev', ydev_tr))}
        probe_labels = {'act': yact_tr, 'dev': ydev_tr}
        act_head_params = list(model.activity_head.parameters()) if use_act else []
        dev_head_params = list(model.device_head.parameters()) if use_dev else []

    restart_every = int(getattr(args, 'disc_restart_every', 0) or 0)

    for epoch in range(args.epochs):
        if restart_every and epoch > 0 and epoch % restart_every == 0:
            # Standard anti-collapse fix: periodically reinitialise the discriminator
            # head(s) so a head that has settled into a degenerate constant predictor
            # cannot simply stay there. Weights are reset in place (reset_parameters
            # keeps the same Parameter objects, only the values change); the optimiser
            # is rebuilt so stale Adam momentum from the old weights is not applied to
            # the fresh ones, and the LR schedule's phase is preserved via last_epoch.
            if use_act:
                for m in model.activity_head.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            if use_dev:
                for m in model.device_head.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            opt, sched = make_opt()
            for _ in range(epoch):
                sched.step()

        model.train()
        if mode == 'adversarial':
            p = epoch / max(1, args.epochs)
            model.set_lambdas(lam_max * (2. / (1. + np.exp(-10 * p)) - 1))
            lam_iidm = lam_max if getattr(args, 'iidm_constant_lambda', False) else \
                lam_max * (2. / (1. + np.exp(-10 * p)) - 1)

        last_epoch = (epoch == args.epochs - 1)
        a_ok = d_ok = n_seen = 0
        if track:
            lam_now = (float(model.grl_act.lambda_) if (mode == 'adversarial' and use_act)
                       else 0.0)
            lr_b, lr_h = opt.param_groups[0]['lr'], opt.param_groups[1]['lr']
            S = dict(id_loss=0., act_loss=0., dev_loss=0., gn_enc=0., gn_h_act=0.,
                     gn_h_dev=0., gn_enc_id=0., gn_enc_adv_act=0., gn_enc_adv_dev=0.,
                     nb=0, ns=0)

        for bi, (xb, ys, ya, yd) in enumerate(loader):
            xb, ys, ya, yd = xb.to(dev_torch), ys.to(dev_torch), ya.to(dev_torch), yd.to(dev_torch)
            opt.zero_grad()
            if iidm:
                f_b = model.embed(xb)
                io = model.identity_classifier(f_b)
                f_a = f_b.detach().requires_grad_(iidm_gp > 0)
                f_d = f_b.detach().requires_grad_(iidm_gp > 0)
                ao = model.activity_head(f_a) if use_act else None
                do = model.device_head(f_d) if use_dev else None
            else:
                io, ao, do = model(xb)
            l_id = ce(io, ys)
            loss = l_id
            l_act = l_dev = None
            if use_act:
                l_act = ce(ao, ya)
                loss = loss + l_act
                if iidm:
                    loss = loss + lam_iidm * _iidm_match(model.activity_head, f_b, ya, Q_act, iidm_gamma)
                    if iidm_gp > 0:
                        loss = loss + 0.5 * iidm_gp * _r1_penalty(ao, f_a, ya)
            if use_dev:
                l_dev = ce(do, yd)
                loss = loss + l_dev
                if iidm:
                    loss = loss + lam_iidm * _iidm_match(model.device_head, f_b, yd, Q_dev, iidm_gamma)
                    if iidm_gp > 0:
                        loss = loss + 0.5 * iidm_gp * _r1_penalty(do, f_d, yd)
            if track and bi % 8 == 0:
                # Split the encoder's gradient into the identity part and the
                # (reversed, lambda-scaled) adversarial part, on a subsample of batches.
                S['gn_enc_id'] += _grad_norm_of(l_id, backbone_params)
                if use_act:
                    S['gn_enc_adv_act'] += _grad_norm_of(l_act, backbone_params)
                if use_dev:
                    S['gn_enc_adv_dev'] += _grad_norm_of(l_dev, backbone_params)
                S['ns'] += 1
            loss.backward()
            if track:
                S['gn_enc'] += _grad_norm_stored(backbone_params)
                S['gn_h_act'] += _grad_norm_stored(act_head_params)
                S['gn_h_dev'] += _grad_norm_stored(dev_head_params)
                S['id_loss'] += float(l_id)
                S['act_loss'] += float(l_act) if use_act else 0.
                S['dev_loss'] += float(l_dev) if use_dev else 0.
                S['nb'] += 1
            opt.step()

            if inner_steps > 1 and (use_act or use_dev):
                # extra discriminator-only steps: the embedding is detached and passes no
                # gradient reversal, and the backbone grads are None after zero_grad, so
                # Adam leaves the encoder untouched (state and weights).
                with torch.no_grad():
                    f_det = model.embed(xb)
                for _ in range(inner_steps - 1):
                    opt.zero_grad()
                    l_in = 0.
                    if use_act:
                        l_in = l_in + ce(model.activity_head(f_det), ya)
                    if use_dev:
                        l_in = l_in + ce(model.device_head(f_det), yd)
                    l_in.backward()
                    opt.step()
                opt.zero_grad()

            if last_epoch or track:
                n_seen += xb.size(0)
                if use_act:
                    a_ok += (ao.argmax(1) == ya).sum().item()
                if use_dev:
                    d_ok += (do.argmax(1) == yd).sum().item()
            if last_epoch:
                # confusion counts of the discriminators in the final epoch: they show
                # whether "chance accuracy" is a constant predictor or a spread of
                # wrong predictions (rows = true class, columns = predicted class)
                if use_act:
                    np.add.at(conf_act, (ya.cpu().numpy(), ao.argmax(1).cpu().numpy()), 1)
                if use_dev:
                    np.add.at(conf_dev, (yd.cpu().numpy(), do.argmax(1).cpu().numpy()), 1)
        sched.step()

        if track:
            nb, ns = max(S['nb'], 1), max(S['ns'], 1)
            rec = dict(epoch=epoch + 1, lam=lam_now, lr_backbone=lr_b, lr_head=lr_h,
                       id_loss=S['id_loss'] / nb, act_loss=S['act_loss'] / nb,
                       dev_loss=S['dev_loss'] / nb,
                       act_acc=(a_ok / n_seen) if (use_act and n_seen) else float('nan'),
                       dev_acc=(d_ok / n_seen) if (use_dev and n_seen) else float('nan'),
                       gn_enc=S['gn_enc'] / nb, gn_head_act=S['gn_h_act'] / nb,
                       gn_head_dev=S['gn_h_dev'] / nb, gn_enc_id=S['gn_enc_id'] / ns,
                       gn_enc_adv_act=S['gn_enc_adv_act'] / ns,
                       gn_enc_adv_dev=S['gn_enc_adv_dev'] / ns)
            with torch.no_grad():
                rec['dtheta_enc'] = float(sum(float(((a_ - b_) ** 2).sum()) for a_, b_ in
                                              zip(backbone_params, theta_prev)) ** 0.5)
                rec['dtheta_head'] = float(sum(float(((a_ - b_) ** 2).sum()) for a_, b_ in
                                               zip(head_params, head_prev)) ** 0.5)
                theta_prev = [p_.detach().clone() for p_ in backbone_params]
                head_prev = [p_.detach().clone() for p_ in head_params]
                model.eval()
                emb_fix = model.embed(X_fix)
                rec['rep_drift'] = (float((emb_fix - prev_emb).norm(dim=1).mean()
                                          / prev_emb.norm(dim=1).mean())
                                    if prev_emb is not None else float('nan'))
                prev_emb = emb_fix
            if (epoch + 1) in probe_epochs:
                _, emb_tr = predict(model, Xtr_n, y_subj[train_idx], args.batch_size, dev_torch)
                for k in ('act', 'dev'):
                    fm, lab = probe_masks[k], probe_labels[k]
                    rec['probe_' + k] = probe_accuracy(emb_tr[fm], lab[fm],
                                                       emb_tr[~fm], lab[~fm])
            dyn.append(rec)

        if last_epoch and n_seen:
            # Raw correct/n alongside the ratio, so a binomial test against
            # chance can be run afterward (Sec. 7's significance check) —
            # "the discriminator reads near chance" and "the discriminator is
            # not statistically distinguishable from chance" are different
            # claims, and only the second one needs these counts.
            if use_act:
                diag['disc_act_acc'] = a_ok / n_seen
                diag['disc_act_correct'], diag['disc_act_n'] = a_ok, n_seen
                diag['disc_act_conf'] = conf_act
            if use_dev:
                diag['disc_dev_acc'] = d_ok / n_seen
                diag['disc_dev_correct'], diag['disc_dev_n'] = d_ok, n_seen
                diag['disc_dev_conf'] = conf_dev

    if track:
        diag['dynamics'] = dyn
        if use_act:
            diag['recovery'] = _recovery_curves(model, Xtr_n, y_subj[train_idx], yact_tr,
                                                seg_tr, args, dev_torch, seed)
    return model, mean, std, diag


@torch.no_grad()
def predict(model, X_norm, y_subj, batch_size, dev_torch):
    model.eval()
    loader = DataLoader(EvalDataset(X_norm, y_subj), batch_size=batch_size, shuffle=False)
    preds, embs = [], []
    for xb, _ in loader:
        xb = xb.to(dev_torch)
        f = model.embed(xb)
        preds.append(model.identity_classifier(f).argmax(1).cpu().numpy())
        embs.append(f.cpu().numpy())
    return np.concatenate(preds), np.concatenate(embs)


def predict_head(model, head_name, X_norm, batch_size, dev_torch):
    """Run the trained discriminator HEAD ITSELF (activity_head or device_head), not a
    fresh post-hoc probe, on arbitrary windows. Used to score the discriminator on
    held-out data, the same windows the post-hoc probe is scored on, so the two are
    compared on equal footing rather than training accuracy vs. held-out accuracy."""
    model.eval()
    head = getattr(model, head_name)
    dummy_y = np.zeros(len(X_norm), dtype=np.int64)
    loader = DataLoader(EvalDataset(X_norm, dummy_y), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev_torch)
            f = model.embed(xb)
            preds.append(head(f).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def score_cell(y_true, y_pred, seg_ids, return_detail=False):
    """Window accuracy, balanced accuracy, and bout-level rank-1 (majority vote).

    return_detail=True additionally returns a per-segment breakdown (true label,
    majority vote, whether it was correct, and the raw window accuracy within
    that segment) — the trace needed to explain a coarse cell's number rather
    than just report it. A cell built from a handful of segments moves in
    increments of 1/n_bouts and can land on the same value across many models
    if the same segment(s) are the ones flipping; without this, that looks like
    suspiciously exact agreement rather than what it is.
    """
    if len(y_true) == 0:
        agg = dict(window_acc=float('nan'), balanced_acc=float('nan'),
                   bout_acc=float('nan'), n_windows=0, n_bouts=0)
        return (agg, []) if return_detail else agg

    window_acc = float((y_true == y_pred).mean())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bal = float(balanced_accuracy_score(y_true, y_pred))

    correct = total = 0
    detail = []
    for s in np.unique(seg_ids):
        m = seg_ids == s
        votes = y_pred[m]
        true = int(y_true[m][0])
        vote = int(np.bincount(votes).argmax())
        ok = vote == true
        correct += int(ok)
        total += 1
        if return_detail:
            detail.append(dict(seg_id=int(s), n_windows=int(m.sum()), y_true=true,
                               y_pred_vote=vote, correct=bool(ok),
                               window_acc=float((votes == true).mean())))

    agg = dict(window_acc=window_acc, balanced_acc=bal,
              bout_acc=correct / max(total, 1),
              n_windows=int(len(y_true)), n_bouts=int(total))
    return (agg, detail) if return_detail else agg


def segment_stratified_split(seg_ids, labels, frac=0.7, seed=0):
    """
    Boolean 'fit' mask over windows, split at SEGMENT level and stratified by
    `labels`, so the probe's test half always contains every class and never
    shares raw samples with its training half.
    """
    rng = np.random.RandomState(seed)
    uniq = np.unique(seg_ids)
    seg_label = {int(s): labels[seg_ids == s][0] for s in uniq}
    fit_segs = []
    for lab in np.unique(list(seg_label.values())):
        ss = np.array([s for s in uniq if seg_label[int(s)] == lab])
        rng.shuffle(ss)
        if len(ss) == 1:
            fit_segs.extend(ss.tolist())
        else:
            k = min(max(1, int(round(frac * len(ss)))), len(ss) - 1)
            fit_segs.extend(ss[:k].tolist())
    return np.isin(seg_ids, fit_segs)


def probe_accuracy(emb_train, lab_train, emb_test, lab_test, return_counts=False):
    """
    Linear probe: how much of a factor survives in the frozen embedding.
    This is the x-axis of the Pareto curve — if it does not move as lambda
    rises, the adversary is not actually doing anything.

    return_counts=True additionally returns (n_correct, n_total) on RAW
    (unweighted) accuracy over the probe's held-out test windows — distinct
    from the balanced accuracy reported everywhere else, kept only so a
    binomial test against chance can be run on independent trials (one
    window, one outcome) rather than on the class-averaged recall figure.
    """
    # Fewer than two classes on either side makes balanced accuracy meaningless
    # (it silently reports 0.0 rather than chance) — report nan instead.
    if len(np.unique(lab_train)) < 2 or len(emb_test) == 0 or len(np.unique(lab_test)) < 2:
        return (float('nan'), 0, 0) if return_counts else float('nan')
    mu = emb_train.mean(0)
    sd = emb_train.std(0) + 1e-8
    clf = LogisticRegression(max_iter=3000)
    clf.fit((emb_train - mu) / sd, lab_train)
    pred = clf.predict((emb_test - mu) / sd)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bal = float(balanced_accuracy_score(lab_test, pred))
    if not return_counts:
        return bal
    return bal, int((pred == lab_test).sum()), int(len(lab_test))


def probe_accuracy_knn(emb_train, lab_train, emb_test, lab_test, k=15, seed=0,
                       max_fit=20000, max_test=20000, return_pred=False):
    """
    k-nearest-neighbour probe (k=15, Euclidean on standardised embeddings,
    uniform vote) — a second nonlinear family, with a different inductive bias
    from the MLP: it recovers a factor that is locally clustered in the
    representation even where no smooth global decision boundary exists.
    Fit and test rows are subsampled to at most max_fit / max_test windows
    (the fit/test split is by segment, so neighbours never come from the test
    segment); distances are computed on the GPU when one is available.
    Same guard as the other probes: fewer than two classes -> nan.
    """
    if len(np.unique(lab_train)) < 2 or len(emb_test) == 0 or len(np.unique(lab_test)) < 2:
        return float('nan')
    rng = np.random.RandomState(seed)
    if len(emb_train) > max_fit:
        sel = rng.choice(len(emb_train), max_fit, replace=False)
        emb_train, lab_train = emb_train[sel], lab_train[sel]
    if len(emb_test) > max_test:
        sel = rng.choice(len(emb_test), max_test, replace=False)
        emb_test, lab_test = emb_test[sel], lab_test[sel]
    mu = emb_train.mean(0)
    sd = emb_train.std(0) + 1e-8
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.tensor((emb_train - mu) / sd, dtype=torch.float32, device=dev)
    B = torch.tensor((emb_test - mu) / sd, dtype=torch.float32, device=dev)
    labs = torch.tensor(lab_train, dtype=torch.long, device=dev)
    n_cls = int(max(lab_train.max(), lab_test.max())) + 1
    k = min(k, len(A))
    preds = []
    with torch.no_grad():
        for i in range(0, len(B), 2048):
            nn_idx = torch.cdist(B[i:i + 2048], A).topk(k, largest=False).indices
            votes = torch.nn.functional.one_hot(labs[nn_idx], n_cls).sum(1)
            preds.append(votes.argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    if return_pred:                       # only aligned with lab_test when the test cell was not subsampled
        return pred
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(balanced_accuracy_score(lab_test, pred))


def probe_accuracy_mlp(emb_train, lab_train, emb_test, lab_test, seed=0, max_fit=40000):
    """
    Nonlinear probe: a small MLP (one hidden layer, 64 units) in place of the
    logistic-regression probe, fitted post-hoc on the same frozen embedding
    and the same fit/test split. This is the stricter test Sec. 10.1 calls
    for — a linear probe can only report what is linearly decodable, so a
    factor it misses could still be present nonlinearly. It costs no training
    time (the encoder is already frozen); only the probe fit is nonlinear.
    Same guard as the linear probe: fewer than two classes either side makes
    balanced accuracy meaningless, so return nan rather than a fake 0.0.
    """
    if len(np.unique(lab_train)) < 2 or len(emb_test) == 0 or len(np.unique(lab_test)) < 2:
        return float('nan')
    if len(emb_train) > max_fit:              # keep the fit tractable on the large corpora
        sel = np.random.RandomState(seed).choice(len(emb_train), max_fit, replace=False)
        emb_train, lab_train = emb_train[sel], lab_train[sel]
    mu = emb_train.mean(0)
    sd = emb_train.std(0) + 1e-8
    clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500,
                        early_stopping=True, n_iter_no_change=15,
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # sklearn ConvergenceWarning
        clf.fit((emb_train - mu) / sd, lab_train)
        pred = clf.predict((emb_test - mu) / sd)
        return float(balanced_accuracy_score(lab_test, pred))


def probe_capacity_grid(e_fit, l_fit, e_test, l_test, seed=0):
    """Probe-generalisation-gap study: held-out balanced accuracy of probes of different capacity, and of the
    linear and MLP probes fitted on subsets of the fit set (probe training-data size). Same standardisation,
    fit set and test cell as the headline probes."""
    out = {}
    mu, sd = e_fit.mean(0), e_fit.std(0) + 1e-8
    Xf, Xt = (e_fit - mu) / sd, (e_test - mu) / sd
    rng = np.random.RandomState(seed)

    def bal(clf, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(X, y)
            return float(balanced_accuracy_score(l_test, clf.predict(Xt)))

    for C in (1e-3, 1e-1, 1e1):                                   # linear probe regularisation
        out[f'lr_C{C:g}'] = bal(LogisticRegression(C=C, max_iter=3000), Xf, l_fit)
    for w in (4, 16, 64, 256, 1024):                              # MLP width
        out[f'mlp_w{w}'] = bal(MLPClassifier(hidden_layer_sizes=(w,), alpha=1e-3, max_iter=500, early_stopping=True,
                                             n_iter_no_change=15, random_state=seed), Xf, l_fit)
    out['mlp_256x256'] = bal(MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-3, max_iter=500, early_stopping=True,
                                           n_iter_no_change=15, random_state=seed), Xf, l_fit)
    for n in (100, 300, 1000, 3000):                              # probe fit-set size
        if n >= len(Xf):
            continue
        sel = rng.choice(len(Xf), n, replace=False)
        if len(np.unique(l_fit[sel])) < 2:
            continue
        out[f'fit{n}_lr'] = bal(LogisticRegression(max_iter=3000), Xf[sel], l_fit[sel])
        out[f'fit{n}_mlp'] = bal(MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, early_stopping=False,
                                               random_state=seed), Xf[sel], l_fit[sel])
    for k in (1, 5, 50):                                          # k-NN neighbourhood (15 is the headline probe)
        out[f'knn_k{k}'] = probe_accuracy_knn(e_fit, l_fit, e_test, l_test, k=k, seed=seed)
    # attacker optimisation budget: the 64-unit MLP trained for a fixed number of epochs, no early stopping
    for ep in (5, 20, 100):
        out[f'mlpbudget_e{ep}'] = bal(MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=ep, early_stopping=False,
                                                    random_state=seed), Xf, l_fit)
    # attacker restarts: five random initialisations of the same MLP; best and worst
    rs = [bal(MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, early_stopping=True, n_iter_no_change=15,
                            random_state=100 + i), Xf, l_fit) for i in range(5)]
    out['mlprestart_min'], out['mlprestart_max'] = float(min(rs)), float(max(rs))
    return out


def profile_preds(e_fit, l_fit, e_test, l_test, seed=0):
    """Predictions of the three headline probes (LR C=1, 64-unit MLP, k-NN k=15) on a test cell, for the segment bootstrap
    of the recoverability profile. Same settings as probe_accuracy / probe_accuracy_mlp / probe_accuracy_knn."""
    mu, sd = e_fit.mean(0), e_fit.std(0) + 1e-8
    Xf, Xt = (e_fit - mu) / sd, (e_test - mu) / sd
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr = LogisticRegression(max_iter=3000).fit(Xf, l_fit).predict(Xt)
        ml = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, early_stopping=True, n_iter_no_change=15,
                           random_state=seed).fit(Xf, l_fit).predict(Xt)
    kn = probe_accuracy_knn(e_fit, l_fit, e_test, l_test, k=15, seed=seed, max_test=10 ** 9, return_pred=True)
    return {'lr': lr, 'mlp': ml, 'knn': kn}


def segment_bootstrap(seg, lab, probe_preds, disc_pred, n_boot=500, seed=0):
    """95% interval over test SEGMENTS (resampled with replacement) for the balanced accuracy of the strongest of the
    probes, of the discriminator head, and of G = strongest probe - discriminator, all on the same resampled windows."""
    classes = np.unique(lab)
    segs, sidx = np.unique(seg, return_inverse=True)
    S, K = len(segs), len(classes)
    cl = np.searchsorted(classes, lab)
    n = np.zeros((S, K))
    np.add.at(n, (sidx, cl), 1)
    models = list(probe_preds.values()) + [disc_pred]
    corr = []
    for p in models:
        c = np.zeros((S, K))
        np.add.at(c, (sidx, cl), (p == lab).astype(float))
        corr.append(c)
    rng = np.random.RandomState(seed)
    res = np.zeros((n_boot, len(models)))
    for b in range(n_boot):
        ii = rng.randint(0, S, S)
        N = n[ii].sum(0)
        ok = N > 0
        for m, c in enumerate(corr):
            res[b, m] = (c[ii].sum(0)[ok] / N[ok]).mean()
    best = res[:, :-1].max(1)
    disc = res[:, -1]
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {'probe': q(best), 'disc': q(disc), 'G': q(best - disc)}


# =============================================================================
# Experiments
# =============================================================================
def evaluate_everything(model, mean, std, X, y_subj, y_act, y_dev, seg_id,
                        train_idx, cells, args, dev_torch, collect_detail=False,
                        seed=0):
    out = {'cells': {}}
    if collect_detail:
        out['cell_detail'] = {}

    for name, idx in cells.items():
        if len(idx) == 0:
            res = score_cell(np.array([]), np.array([]), np.array([]),
                             return_detail=collect_detail)
        else:
            preds, _ = predict(model, (X[idx] - mean) / std, y_subj[idx],
                               args.batch_size, dev_torch)
            res = score_cell(y_subj[idx], preds, seg_id[idx], return_detail=collect_detail)
        if collect_detail:
            out['cells'][name], out['cell_detail'][name] = res
        else:
            out['cells'][name] = res

    # --- linear probes: how much activity/device information survives -------
    # Split the training pool BY SEGMENT (so overlapping windows cannot leak into
    # the probe's own test half) and stratify by the factor being probed (so both
    # halves are guaranteed to contain every class).
    _, emb_train = predict(model, (X[train_idx] - mean) / std,
                           y_subj[train_idx], args.batch_size, dev_torch)
    segs_tr = seg_id[train_idx]

    cap = int(getattr(args, 'probe_fit_cap', 0) or 0)

    def capped(emb, lab):
        """same fit-set size for every condition when --probe_fit_cap is set"""
        if cap and len(emb) > cap:
            sel = np.random.RandomState(seed).choice(len(emb), cap, replace=False)
            return emb[sel], lab[sel]
        return emb, lab

    for key, labels in (('probe_activity', y_act[train_idx]),
                        ('probe_device', y_dev[train_idx])):
        fit_mask = segment_stratified_split(segs_tr, labels)
        e_fit, l_fit = capped(emb_train[fit_mask], labels[fit_mask])
        out[key], out[key + '_correct'], out[key + '_n'] = probe_accuracy(
            e_fit, l_fit, emb_train[~fit_mask], labels[~fit_mask], return_counts=True)
        if getattr(args, 'nonlinear_probe', False):
            out[key + '_mlp'] = probe_accuracy_mlp(
                e_fit, l_fit, emb_train[~fit_mask], labels[~fit_mask], seed=seed)
        if getattr(args, 'knn_probe', False):
            out[key + '_knn'] = probe_accuracy_knn(
                e_fit, l_fit, emb_train[~fit_mask], labels[~fit_mask], seed=seed)

    # --- probes scored on windows the encoder never saw ------------------------
    # Fit on the training pool, score on an evaluation cell. A factor is scored on a
    # cell only if the cell's labels for it are all training classes (a held-out
    # device or activity has no label in the probe's label space).
    _prof = {}
    if getattr(args, 'test_cell_probe', False):
        for key, y_all in (('probe_activity', y_act), ('probe_device', y_dev)):
            lab_pool = y_all[train_idx]
            e_fit, l_fit = capped(emb_train, lab_pool)
            for cname in ('seen_seen', 'unseen_dev_seen_act', 'seen_dev_unseen_act', 'unseen_subject'):
                idx = cells.get(cname, [])
                if len(idx) == 0:
                    continue
                lab_c = y_all[idx]
                keep = lab_c >= 0                       # -1 marks the held-out class
                if keep.sum() == 0 or len(np.unique(lab_c[keep])) < 2:
                    continue
                _, emb_c = predict(model, (X[idx[keep]] - mean) / std, y_subj[idx[keep]],
                                   args.batch_size, dev_torch)
                lab_k = lab_c[keep]
                tag = f'{key}_tc_{cname}'
                out[tag] = probe_accuracy(e_fit, l_fit, emb_c, lab_k)
                if getattr(args, 'nonlinear_probe', False):
                    out[tag.replace(key, key + '_mlp', 1)] = probe_accuracy_mlp(
                        e_fit, l_fit, emb_c, lab_k, seed=seed)
                if getattr(args, 'knn_probe', False):
                    out[tag.replace(key, key + '_knn', 1)] = probe_accuracy_knn(
                        e_fit, l_fit, emb_c, lab_k, seed=seed)
                if getattr(args, 'probe_grid', False):
                    if cname == 'seen_seen':
                        for gname, gval in probe_capacity_grid(e_fit, l_fit, emb_c, lab_k, seed=seed).items():
                            out[f'{key}_tc_{cname}_grid_{gname}'] = gval
                    # predictions kept for the segment bootstrap against the discriminator head (below)
                    _prof[(key, cname)] = dict(seg=seg_id[idx[keep]], lab=lab_k,
                                               preds=profile_preds(e_fit, l_fit, emb_c, lab_k, seed=seed))
                if cname == 'seen_seen' and getattr(args, 'split_probe', False):
                    # Same fitted probes, scored separately on two disjoint halves of the evaluation cell
                    # (split by segment, stratified by label, fixed seed so the halves are the same at every
                    # lambda and seed). Half A is for CHOOSING lambda, half B for REPORTING, so the reported
                    # reduction does not use the data that picked the minimum (review: lambda-selection bias).
                    half_a = segment_stratified_split(seg_id[idx[keep]], lab_k, frac=0.5, seed=7)
                    for hname, m in (('A', half_a), ('B', ~half_a)):
                        if m.sum() == 0 or len(np.unique(lab_k[m])) < 2:
                            continue
                        out[f'{tag}_half{hname}'] = probe_accuracy(e_fit, l_fit, emb_c[m], lab_k[m])
                        if getattr(args, 'nonlinear_probe', False):
                            out[tag.replace(key, key + '_mlp', 1) + f'_half{hname}'] = probe_accuracy_mlp(
                                e_fit, l_fit, emb_c[m], lab_k[m], seed=seed)
                        if getattr(args, 'knn_probe', False):
                            out[tag.replace(key, key + '_knn', 1) + f'_half{hname}'] = probe_accuracy_knn(
                                e_fit, l_fit, emb_c[m], lab_k[m], seed=seed)

    # --- the discriminator HEAD ITSELF, scored on windows the encoder never saw -----
    # Discriminator accuracy is normally training accuracy (Sec. 3.6); this scores the
    # same trained head on the same held-out cells the probe above uses, so the two are
    # compared on equal footing instead of training accuracy vs. held-out accuracy.
    if getattr(args, 'disc_heldout', False):
        for key, y_all, head_name in (
                ('disc_act', y_act, 'activity_head'),
                ('disc_dev', y_dev, 'device_head')):
            if not hasattr(model, head_name):
                continue
            for cname in ('seen_seen', 'unseen_dev_seen_act', 'seen_dev_unseen_act', 'unseen_subject'):
                idx = cells.get(cname, [])
                if len(idx) == 0:
                    continue
                lab_c = y_all[idx]
                keep = lab_c >= 0
                if keep.sum() == 0 or len(np.unique(lab_c[keep])) < 2:
                    continue
                preds = predict_head(model, head_name, (X[idx[keep]] - mean) / std,
                                     args.batch_size, dev_torch)
                lab_k = lab_c[keep]
                tag = f'{key}_heldout_tc_{cname}'
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    out[tag] = float(balanced_accuracy_score(lab_k, preds))
                out[tag + '_correct'] = int((preds == lab_k).sum())
                out[tag + '_n'] = int(len(lab_k))
                pk = _prof.get(('probe_' + key.split('_')[1].replace('act', 'activity').replace('dev', 'device'), cname))
                if pk is not None and len(pk['lab']) == len(lab_k):
                    bt = segment_bootstrap(pk['seg'], lab_k, pk['preds'], preds, seed=seed)
                    pt = tag.replace('disc_act', 'probe_activity').replace('disc_dev', 'probe_device').replace('_heldout', '')
                    out[pt + '_boot_G_lo'], out[pt + '_boot_G_hi'] = bt['G']
                    out[pt + '_boot_probe_lo'], out[pt + '_boot_probe_hi'] = bt['probe']
                    out[pt + '_boot_disc_lo'], out[pt + '_boot_disc_hi'] = bt['disc']
                    out[pt + '_majority_raw'] = float(np.bincount(lab_k).max() / len(lab_k))
    return out


def aggregate(runs):
    """Mean/std across seeds for a list of evaluate_everything() dicts."""
    agg = {'cells': {}}
    for cell in runs[0]['cells']:
        agg['cells'][cell] = {}
        for metric in runs[0]['cells'][cell]:
            vals = [r['cells'][cell][metric] for r in runs]
            # Empty cells (single-nuisance datasets) are all-nan by design, not a
            # problem — suppress numpy's warning rather than printing it per lambda.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                agg['cells'][cell][metric] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    for k in ('probe_activity', 'probe_device', 'disc_act_acc', 'disc_dev_acc',
             'probe_activity_mlp', 'probe_device_mlp',
             'probe_activity_knn', 'probe_device_knn'):
        vals = [r.get(k, float('nan')) for r in runs]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        # Raw per-seed values alongside the (mean, std) tuple above — needed
        # for a paired test across seeds (Sec. 7's significance check), which
        # a summary statistic cannot support.
        agg[k + '_raw'] = vals

    # Raw correct/n counts, per seed, for the probe-vs-chance binomial test.
    # Not meaningfully summarised as mean/std (they are counts, not rates),
    # so kept only as per-seed lists.
    for k in ('probe_activity_correct', 'probe_activity_n',
             'probe_device_correct', 'probe_device_n',
             'disc_act_correct', 'disc_act_n', 'disc_dev_correct', 'disc_dev_n',
             'disc_act_conf', 'disc_dev_conf'):
        agg[k + '_raw'] = [r.get(k) for r in runs]

    # probes scored on the evaluation cells (--test_cell_probe): per-seed values
    tc_keys = sorted({k for r in runs for k in r if '_tc_' in k})
    for k in tc_keys:
        vals = [r.get(k, float('nan')) for r in runs]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            agg[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        agg[k + '_raw'] = vals
    return agg


def run(args):
    dev_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", dev_torch)

    # In auxiliary mode there is no gradient-reversal layer, so lambda never reaches
    # anything — sweeping it just retrains one configuration N times.
    if args.mode == 'auxiliary' and args.experiment == 'sweep' and len(args.lambda_sweep) > 1:
        print("\n  NOTE: --mode auxiliary has no gradient reversal, so lambda is inert. "
              "Collapsing\n  the sweep to a single run. Compare this one number against "
              "adversarial lambda=0.\n")
        args.lambda_sweep = [0.0]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.npz:
        import json as _json
        print(f"Loading windowed dataset from {args.npz}")
        d = np.load(args.npz, allow_pickle=True)
        X = d['X']
        y_subject, y_activity, y_device = d['y_subject'], d['y_activity'], d['y_device']
        seg_id = d['seg_id']
        ds_meta = _json.loads(str(d['meta'])) if 'meta' in d.files else {}
        ds_name = ds_meta.get('dataset', os.path.splitext(os.path.basename(args.npz))[0])
        nuis_name = ds_meta.get('nuisance_name', 'device')
        print(f"  dataset={ds_name}  nuisance={nuis_name}  X={X.shape}  "
              f"segments={len(np.unique(seg_id))}")
        print(f"  subjects({len(np.unique(y_subject))}) activities"
              f"({len(np.unique(y_activity))}) nuisance({len(np.unique(y_device))})")
        return _run_experiments(args, X, y_subject, y_activity, y_device, seg_id,
                                ds_name, nuis_name)

    if not args.hhar_zip:
        raise SystemExit("Provide either --hhar_zip (HHAR) or --npz (any dataset "
                         "built by har_datasets.py).")

    extracted = extract_hhar(args.hhar_zip, os.path.join(args.out_dir, 'hhar_extracted'))
    acc_path, gyro_path = locate_phone_csvs(extracted)

    if args.diagnose:
        df = load_phone_csv(acc_path, 'phone accelerometer')
        diagnose_rates(df, 'Phones_accelerometer.csv')
        diagnose_coverage(df)
        print("\nDiagnostic complete. Set --target_hz at or below the slowest device, "
              "then re-run without --diagnose.")
        return

    act_tag = 'all' if not args.activities else '-'.join(sorted(args.activities))
    ex_tag = 'none' if not args.exclude_devices else '-'.join(sorted(args.exclude_devices))
    cache = os.path.join(
        args.out_dir,
        f'windows_hz{args.target_hz:g}_win{args.window_seconds:g}_ov{args.overlap:g}'
        f'_{act_tag}_ex-{ex_tag}.npz')
    X, y_subject, y_activity, y_device, seg_id = build_windowed_dataset(
        acc_path, gyro_path, args, cache)

    return _run_experiments(args, X, y_subject, y_activity, y_device, seg_id,
                            'hhar', 'device', act_tag=act_tag, ex_tag=ex_tag)


def _atomic_dump(obj, path):
    """Write a pickle atomically (temp file, then rename), so a process killed
    mid-write — hibernation, a closed session — cannot leave a corrupt checkpoint
    that the resume logic would then fail to load."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def degrade_windows(X, y_activity, held_out_activity, mode, level, seed=12345):
    """
    Controlled degradation of activity separability, on IDENTICAL data.

    Every subject, recording, segment, window, activity proportion and evaluation
    cell is unchanged; only the signal in each window is altered, once, with a
    fixed random seed (a data manipulation, not augmentation: every training run
    sees the same degraded windows).
      noise  X += level * (per-channel std) * N(0, 1), independently per window,
             applied to all windows. Lowers the separability of every factor.
      mix    each window of a SEEN activity is blended with a random window of a
             DIFFERENT seen activity: x = (1 - level) * x + level * x_other, label
             unchanged. level=0.5 makes the two classes indistinguishable in
             expectation. Controlled class overlap; windows of the held-out
             activity are left untouched so the held-out cells are unchanged.
    """
    rng = np.random.RandomState(seed)
    X = X.astype(np.float32, copy=True)
    if mode == 'noise':
        sd = X.std(axis=(0, 1), keepdims=True)
        X += (level * sd * rng.standard_normal(X.shape)).astype(np.float32)
    elif mode == 'mix':
        seen = np.where(y_activity != held_out_activity)[0]
        src = X.copy()
        for c in np.unique(y_activity[seen]):
            own = seen[y_activity[seen] == c]
            other = seen[y_activity[seen] != c]
            pick = rng.choice(other, size=len(own), replace=True)
            X[own] = (1.0 - level) * src[own] + level * src[pick]
    else:
        raise ValueError(mode)
    return X


def _run_experiments(args, X, y_subject, y_activity, y_device, seg_id,
                     ds_name, nuis_name, act_tag=None, ex_tag=None):
    dev_torch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if getattr(args, 'degrade', 'none') != 'none':
        X = degrade_windows(X, y_activity, args.held_out_activity, args.degrade,
                            args.degrade_level)
        print(f"\n  Degraded windows: mode={args.degrade} level={args.degrade_level:g} "
              f"(same subjects/segments/windows/cells; only the signal changes)")
    if act_tag is None:
        act_tag = 'all' if not args.activities else '-'.join(sorted(args.activities))
    if ex_tag is None:
        ex_tag = 'none' if not args.exclude_devices else '-'.join(sorted(args.exclude_devices))

    # A dataset with a single nuisance level (MotionSense) cannot hold one out and
    # cannot train a discriminator on it. Disable that branch rather than fitting a
    # one-class head and reporting a meaningless probe.
    single_nuisance = len(np.unique(y_device)) < 2
    if single_nuisance:
        print(f"\n  NOTE: '{ds_name}' has a single {nuis_name} level "
              f"({np.unique(y_device)[0]}). Disabling the nuisance-adversarial branch and\n"
              f"  the cross-{nuis_name} test cells; only the activity axis is testable here.")
        args.held_out_device = '__none__'

    subj_enc = LabelEncoder()
    y_subj = subj_enc.fit_transform(y_subject)
    n_subj = len(subj_enc.classes_)

    seen_acts = sorted(set(y_activity) - {args.held_out_activity})
    seen_devs = sorted(set(y_device) - {args.held_out_device})
    print(f"\nSeen activities: {seen_acts}\nSeen devices: {seen_devs}")
    assert seen_acts and seen_devs, "Held-out choice leaves nothing to train on."

    act_enc = LabelEncoder().fit(seen_acts)
    dev_enc = LabelEncoder().fit(seen_devs)

    def safe_encode(enc, values, held):
        out = np.full(len(values), -1, dtype=np.int64)
        m = values != held
        out[m] = enc.transform(values[m])
        return out

    y_act = safe_encode(act_enc, y_activity, args.held_out_activity)
    y_dev = safe_encode(dev_enc, y_device, args.held_out_device)

    train_idx, cells = build_2d_split(seg_id, y_activity, y_device,
                                      args.held_out_device, args.held_out_activity,
                                      args.holdback_frac)
    held_subjects = []
    if getattr(args, 'hold_subjects_n', 0):
        subs = np.unique(y_subject)
        rs = np.random.RandomState(1000 + args.hold_subjects_seed)
        held_subjects = sorted(rs.choice(subs, size=args.hold_subjects_n, replace=False).tolist())
        hs = np.isin(y_subject, held_subjects)
        train_idx = train_idx[~hs[train_idx]]
        cells = {k: v[~hs[v]] for k, v in cells.items()}
        cells['unseen_subject'] = np.where(hs & (y_device != args.held_out_device) & (y_activity != args.held_out_activity))[0]
        print(f"\n  Held-out SUBJECTS ({len(held_subjects)}): {held_subjects}. They are removed from training, the probe fit "
              f"pool and all other cells; cell 'unseen_subject' holds their windows (seen device, seen activity).")
    if args.max_train_windows and len(train_idx) > args.max_train_windows:
        segs = np.unique(seg_id[train_idx])
        rng = np.random.RandomState(1234)
        rng.shuffle(segs)
        keep, n = [], 0
        for sg in segs:
            k = int((seg_id[train_idx] == sg).sum())
            if n + k > args.max_train_windows and keep:
                break
            keep.append(sg); n += k
        before = len(train_idx)
        train_idx = train_idx[np.isin(seg_id[train_idx], keep)]
        print(f"\n  Subsampled training set: {before:,} -> {len(train_idx):,} windows "
              f"({len(keep)} segments), to match --max_train_windows={args.max_train_windows:,}")

    print(f"\nTrain windows: {len(train_idx):,}  "
          f"({len(np.unique(seg_id[train_idx]))} segments)")
    for k, v in cells.items():
        n_seg = len(np.unique(seg_id[v])) if len(v) else 0
        print(f"  cell '{k}': {len(v):,} windows, {n_seg} segments")
        if 0 < n_seg < 10:
            print(f"      ^ only {n_seg} segments — bout-level accuracy here is coarse "
                  f"(resolution 1/{n_seg}). Report window + balanced accuracy alongside it.")

    # NOTE: every setting that changes the DATA or the TRAINING must appear here.
    # An earlier version omitted the activity subset and the excluded devices, so a
    # run with a different --activities silently resumed a checkpoint from a
    # different configuration and reported stale numbers under a new name.
    seed_tag = '-'.join(str(s) for s in args.seeds)
    tag = (f"{ds_name}_dev-{args.held_out_device}_act-{args.held_out_activity}"
           f"_hz{args.target_hz:g}_win{args.window_seconds:g}"
           f"_acts-{act_tag}_ex-{ex_tag}"
           f"_{args.pooling}_{args.mode}_dlm{args.disc_lr_mult:g}"
           f"_ep{args.epochs}_seeds{seed_tag}"
           f"{'_cap' + str(args.max_train_windows) if args.max_train_windows else ''}"
           f"{'_enc' + args.encoder if args.encoder != 'cnn' else ''}"
           f"{'_dh' + str(args.disc_hidden) if args.disc_hidden != 128 else ''}"
           f"{'_deg' + args.degrade + format(args.degrade_level, 'g') if args.degrade != 'none' else ''}"
           f"{'_hs' + str(args.hold_subjects_n) + 'x' + str(args.hold_subjects_seed) if getattr(args, 'hold_subjects_n', 0) else ''}")
    ckpt = os.path.join(args.out_dir, f'results_{args.experiment}_{tag}.pkl')
    results = pickle.load(open(ckpt, 'rb')) if os.path.exists(ckpt) else {}
    if results:
        print(f"Resumed checkpoint: {list(results.keys())}")

    common = dict(n_subj=n_subj, n_act=len(act_enc.classes_), n_dev=len(dev_enc.classes_))

    if args.experiment == 'sweep':
        segdetail_path = ckpt.replace('.pkl', '_segdetail.pkl')
        segdetail = (pickle.load(open(segdetail_path, 'rb'))
                    if args.segment_diagnostics and os.path.exists(segdetail_path) else {})
        dynamics_path = ckpt.replace('.pkl', '_dynamics.pkl')
        dynamics = (pickle.load(open(dynamics_path, 'rb'))
                    if args.track_dynamics and os.path.exists(dynamics_path) else {})

        for lam in args.lambda_sweep:
            key = f"lambda={lam:g}"
            if key in results:
                print(f"Skipping {key} (checkpointed).")
                continue
            print(f"\n{'#'*70}\n{key}   mode={args.mode}\n{'#'*70}")
            runs = []
            seed_details = []
            dyn_runs = []
            for seed in args.seeds:
                model, mean, std, diag = train_model(
                    X, y_subj, y_act, y_dev, train_idx,
                    common['n_subj'], common['n_act'], common['n_dev'],
                    use_act=True, use_dev=not single_nuisance, args=args, seed=seed,
                    lam_max=lam, mode=args.mode, dev_torch=dev_torch, seg_id=seg_id)
                if args.track_dynamics:
                    dyn_runs.append({'seed': seed, 'dynamics': diag.pop('dynamics', None),
                                     'recovery': diag.pop('recovery', None)})
                r = evaluate_everything(model, mean, std, X, y_subj, y_act, y_dev,
                                        seg_id, train_idx, cells, args, dev_torch,
                                        collect_detail=args.segment_diagnostics, seed=seed)
                if args.segment_diagnostics:
                    seed_details.append({'seed': seed, 'cell_detail': r.pop('cell_detail')})
                r.update(diag)
                runs.append(r)
                print(f"  seed {seed}: " + ", ".join(
                    f"{c}={r['cells'][c]['bout_acc']:.3f}" for c in cells) +
                    f" | probe_act={r['probe_activity']:.3f} probe_dev={r['probe_device']:.3f}"
                    f" | disc_act={diag['disc_act_acc']:.3f} disc_dev={diag['disc_dev_acc']:.3f}")
            results[key] = aggregate(runs)
            _atomic_dump(results, ckpt)
            if args.segment_diagnostics:
                segdetail[key] = seed_details
                _atomic_dump(segdetail, segdetail_path)
            if args.track_dynamics:
                dynamics[key] = dyn_runs
                _atomic_dump(dynamics, dynamics_path)
        print_sweep(results, args, ckpt, common['n_act'], common['n_dev'])
        if args.segment_diagnostics:
            print(f"Segment-level diagnostics -> {segdetail_path}")

    else:
        methods = {
            'baseline': (False, False),
            f'activity_{args.mode}': (True, False),
            f'device_{args.mode}': (False, True),
            f'dual_{args.mode}': (True, True),
        }
        for name, (ua, ud) in methods.items():
            if name in results:
                print(f"Skipping {name} (checkpointed).")
                continue
            print(f"\n{'#'*70}\nMethod: {name}\n{'#'*70}")
            runs = []
            for seed in args.seeds:
                model, mean, std, diag = train_model(
                    X, y_subj, y_act, y_dev, train_idx,
                    common['n_subj'], common['n_act'], common['n_dev'],
                    use_act=ua, use_dev=ud and not single_nuisance, args=args, seed=seed,
                    lam_max=args.lambda_max, mode=args.mode, dev_torch=dev_torch)
                r = evaluate_everything(model, mean, std, X, y_subj, y_act, y_dev,
                                        seg_id, train_idx, cells, args, dev_torch,
                                        seed=seed)
                r.update(diag)
                runs.append(r)
            results[name] = aggregate(runs)
            _atomic_dump(results, ckpt)
        print_methods(results, cells, args)

    return results


# =============================================================================
# Reporting
# =============================================================================
CELLS = ['seen_seen', 'unseen_dev_seen_act', 'seen_dev_unseen_act', 'unseen_unseen']


def print_methods(results, cells, args):
    print(f"\n{'='*100}\nFINAL — bout-level rank-1 (window acc in parentheses)\n{'='*100}")
    print(f"{'Method':<24}" + "".join(f"{c:<20}" for c in CELLS))
    for name, r in results.items():
        row = f"{name:<24}"
        for c in CELLS:
            b = r['cells'][c]['bout_acc'][0]
            w = r['cells'][c]['window_acc'][0]
            row += f"{b:.3f} ({w:.3f})".ljust(20)
        print(row)


def _removable_fraction(p0, p1, chance):
    """
    Fraction of the AVAILABLE information actually removed. A raw probe drop is
    not comparable across factors with different class counts: 0.99 -> 0.95 on a
    5-class factor (chance 0.20) is 4% removed, while 0.84 -> 0.69 on a 2-class
    factor (chance 0.50) is 45% removed. This is the number to report.
    """
    headroom = p0 - chance
    if not np.isfinite(headroom) or headroom <= 1e-6:
        return float('nan')
    return (p0 - p1) / headroom


def _interpret(df, n_act, n_dev):
    """
    Report the BEST operating point, not the endpoint.

    Adversarial invariance is non-monotonic in lambda: pressure that is too strong
    destroys the discriminator, and a dead discriminator supplies no invariance
    signal, so the probes recover while the accuracy cost stays paid. Comparing
    lambda_min against lambda_max therefore reports "nothing happened" for a sweep
    in which a great deal happened somewhere in the middle.
    """
    df = df.sort_values('lambda_max').reset_index(drop=True)
    base = df.iloc[0]

    ident_cell = ('unseen_dev_seen_act_bout'
                  if np.isfinite(base.get('unseen_dev_seen_act_bout', np.nan))
                  else 'seen_dev_unseen_act_bout')

    print(f"\n{'-'*78}\nREADING THE SWEEP\n{'-'*78}")

    best = {}
    for factor, probe_col, n_cls in (('ACTIVITY', 'probe_activity', n_act),
                                     ('DEVICE  ', 'probe_device', n_dev)):
        chance = 1.0 / max(n_cls, 2)
        if not np.isfinite(base[probe_col]):
            print(f"  {factor}  n/a (single level)")
            best[probe_col] = None
            continue
        i = int(df[probe_col].idxmin())
        row = df.iloc[i]
        frac = _removable_fraction(base[probe_col], row[probe_col], chance)
        cost = base[ident_cell] - row[ident_cell]
        print(f"  {factor}  probe {base[probe_col]:.3f} -> {row[probe_col]:.3f} "
              f"at lambda={row['lambda_max']:g}  (chance {chance:.3f})")
        print(f"            removed {frac*100:5.1f}% of available info "
              f"for {cost:+.3f} identity ({ident_cell.replace('_bout','')})")
        if i != len(df) - 1:
            print(f"            NON-MONOTONIC: recovers to {df.iloc[-1][probe_col]:.3f} "
                  f"by lambda={df.iloc[-1]['lambda_max']:g} — beyond the optimum you pay "
                  f"the cost\n            without buying invariance.")
        best[probe_col] = (row['lambda_max'], frac, cost)

    # Discriminator collapse: without this check a recovered probe looks like the
    # factor being irremovable, when in fact the adversary simply died.
    for name, col, n_cls in (('activity', 'disc_act_acc', n_act), ('device', 'disc_dev_acc', n_dev)):
        v = df[col].iloc[-1]
        if np.isfinite(v) and v < 1.5 / max(n_cls, 2):
            print(f"\n  WARNING: the {name} discriminator is at {v:.3f} by the highest lambda "
                  f"(chance {1.0/max(n_cls,2):.3f}).\n  It has collapsed — high-lambda rows "
                  f"measure a broken adversary, not an irremovable factor.")

    print()
    fa, fd = best.get('probe_activity'), best.get('probe_device')
    if fa and fa[1] > 0.30 and abs(fa[2]) < 0.03:
        print(f"  VERDICT: activity information IS removable here ({fa[1]*100:.0f}% at "
              f"lambda={fa[0]:g})\n  at negligible identity cost ({fa[2]:+.3f}). The factors "
              f"ARE separable in this dataset.\n  If another dataset says otherwise, the "
              f"difference is the finding — check how decodable\n  activity was to begin with "
              f"(a near-perfectly decodable factor resists removal).")
    elif fa and fd and (fd[1] - fa[1]) > 0.20 and fa[1] < 0.10:
        print("  VERDICT: ASYMMETRY. The same adversary strips the nuisance factor but barely")
        print("  touches activity, so 'the adversary is broken' is ruled out — activity")
        print("  information is genuinely resistant to removal here.")
    elif fa and fa[1] < 0.10 and fa[2] > 0.02:
        print("  VERDICT: identity degraded without activity being removed — gradient reversal")
        print("  damaging the representation. Raise --disc_lr_mult and re-check disc_act.")
    elif fa and fa[1] >= 0.10:
        print(f"  VERDICT: activity {fa[1]*100:.0f}% removable at lambda={fa[0]:g}, costing "
              f"{fa[2]:+.3f} identity.\n  Report that pair as the operating point.")
    else:
        print("  VERDICT: nothing moved at any lambda. Check the branches are active.")

    # len(df) must be checked before indexing iloc[1] — mode='auxiliary' collapses
    # the sweep to a single row, and `and` does not save a short-circuit here if
    # the index access comes first (it still gets evaluated left-to-right).
    if fa and len(df) > 2 and fa[0] == df['lambda_max'].iloc[1]:
        print(f"\n  The optimum sits at the SECOND-smallest lambda tried. Re-sweep finer below "
              f"it\n  (e.g. --lambda_sweep 0 0.02 0.05 0.1 0.15 0.2 0.3) — the true optimum may "
              f"be lower.")


def print_sweep(results, args, ckpt, n_act=5, n_dev=2):
    print(f"\n{'='*108}")
    print("PARETO SWEEP — gait ID (bout-level rank-1) vs. how much activity/device "
          "information survives")
    print(f"{'='*108}")
    header = (f"{'lambda':<9}" + "".join(f"{c:<16}" for c in CELLS) +
              f"{'probe_act':<11}{'probe_dev':<11}{'disc_act':<10}{'disc_dev':<10}")
    print(header)

    rows = []
    for key in sorted(results, key=lambda k: float(k.split('=')[1])):
        r = results[key]
        lam = float(key.split('=')[1])
        line = f"{lam:<9.3g}"
        for c in CELLS:
            m, s = r['cells'][c]['bout_acc']
            line += f"{m:.3f}±{s:.3f}".ljust(16)
        pa, pa_s = r['probe_activity']
        pd_, pd_s = r['probe_device']
        da, da_s = r.get('disc_act_acc', (float('nan'), float('nan')))
        dd, dd_s = r.get('disc_dev_acc', (float('nan'), float('nan')))
        pa_mlp, pa_mlp_s = r.get('probe_activity_mlp', (float('nan'), float('nan')))
        pd_mlp, pd_mlp_s = r.get('probe_device_mlp', (float('nan'), float('nan')))
        pa_knn, pa_knn_s = r.get('probe_activity_knn', (float('nan'), float('nan')))
        pd_knn, pd_knn_s = r.get('probe_device_knn', (float('nan'), float('nan')))
        line += f"{pa:<11.3f}{pd_:<11.3f}{da:<10.3f}{dd:<10.3f}"
        if np.isfinite(pa_mlp) or np.isfinite(pd_mlp):
            line += f"  mlp_act={pa_mlp:.3f} mlp_dev={pd_mlp:.3f}"
        if np.isfinite(pa_knn) or np.isfinite(pd_knn):
            line += f"  knn_act={pa_knn:.3f} knn_dev={pd_knn:.3f}"
        print(line)
        # Every *_std column below comes straight from aggregate()'s per-seed
        # (mean, std) tuples — the 3-seed variance the console table already
        # shows for bout_acc (as "m±s") but the CSV used to drop for every
        # other metric, including the probe/discriminator pair the paper's
        # central claim rests on. Keep both so a reader can see how noisy the
        # gap between them is, not just its mean.
        rows.append(dict(
            lambda_max=lam,
            **{f'{c}_bout': r['cells'][c]['bout_acc'][0] for c in CELLS},
            **{f'{c}_bout_std': r['cells'][c]['bout_acc'][1] for c in CELLS},
            **{f'{c}_window': r['cells'][c]['window_acc'][0] for c in CELLS},
            **{f'{c}_window_std': r['cells'][c]['window_acc'][1] for c in CELLS},
            **{f'{c}_balanced': r['cells'][c]['balanced_acc'][0] for c in CELLS},
            **{f'{c}_balanced_std': r['cells'][c]['balanced_acc'][1] for c in CELLS},
            probe_activity=pa, probe_activity_std=pa_s,
            probe_device=pd_, probe_device_std=pd_s,
            disc_act_acc=da, disc_act_acc_std=da_s,
            disc_dev_acc=dd, disc_dev_acc_std=dd_s,
            probe_activity_mlp=pa_mlp, probe_activity_mlp_std=pa_mlp_s,
            probe_device_mlp=pd_mlp, probe_device_mlp_std=pd_mlp_s,
            probe_activity_knn=pa_knn, probe_activity_knn_std=pa_knn_s,
            probe_device_knn=pd_knn, probe_device_knn_std=pd_knn_s))

    csv_path = ckpt.replace('.pkl', '_pareto.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nPareto data -> {csv_path}")

    _interpret(pd.DataFrame(rows), n_act, n_dev)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        df = pd.DataFrame(rows).sort_values('lambda_max')
        for c, mk in zip(CELLS, ['o', 's', '^', 'D']):
            ax.plot(df['probe_activity'], df[f'{c}_bout'], marker=mk, label=c)
        for _, rr in df.iterrows():
            ax.annotate(f"λ={rr['lambda_max']:g}",
                        (rr['probe_activity'], rr[f'{CELLS[-1]}_bout']),
                        fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel("Activity information retained (linear-probe balanced acc.)")
        ax.set_ylabel("Gait identification (bout-level rank-1)")
        ax.set_title(f"Identity/activity trade-off — mode={args.mode}, pooling={args.pooling}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = ckpt.replace('.pkl', '_pareto.png')
        fig.savefig(png, dpi=160)
        print(f"Pareto figure -> {png}")
    except Exception as e:                                    # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    run(parse_args())

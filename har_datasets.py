"""
Multi-dataset adapter for IMU identity/activity experiments
============================================================
One interface over HHAR, RealWorld (HAR) 2016, MotionSense and MobiAct v2.0, so
the downstream experiment never learns anything dataset-specific.

Every dataset is reduced to the same tuple:

    X          (N, T, 6)  float32   acc xyz + gyro xyz, resampled to a common rate
    subject    (N,)       str       identity label — the thing being recognised
    activity   (N,)       str       canonical activity vocabulary (see CANONICAL_ACTIVITY)
    nuisance   (N,)       str       the extrinsic factor under test
    seg_id     (N,)       int64     contiguous-recording id; ALL splits go by this

WHAT THE NUISANCE FACTOR IS, PER DATASET
----------------------------------------
    hhar         phone model      (nexus4 / s3 / s3mini / samsungold)   4 levels    9 subj
    realworld    body position    (chest/forearm/head/shin/thigh/...)   7 levels   15 subj
    motionsense  none             single device, single position        1 level    24 subj
    mobiact      none             single device, single position        1 level   ~66 subj

That difference is the point. HHAR tests whether *device* is removable, RealWorld
tests whether *position* is removable, and MotionSense and MobiAct hold every
nuisance factor constant so the activity axis is isolated — MobiAct additionally
at a subject count where chance identity accuracy is ~1.5% instead of ~11%.

MotionSense and MobiAct have one nuisance level, so `has_nuisance` is False in
their metadata and the nuisance-adversarial branch must be skipped for them — the
adapter reports this rather than leaving the experiment to discover it.

MobiAct's four fall events are excluded by default (brief transients, not sustained
activities); pass --include_falls to keep them.

UNITS AND CONVENTIONS (cross-dataset comparability)
---------------------------------------------------
- Accelerometer is GRAVITY-INCLUSIVE m/s^2 everywhere. MotionSense stores
  userAcceleration (gravity removed) and gravity separately in g, so this adapter
  adds them back together and multiplies by 9.80665.
- Gyroscope is rad/s everywhere.
- Activity names are mapped to one vocabulary so 'climbingup', 'ups' and
  'stairsup' become the same label across datasets.

USAGE
-----
    # ALWAYS inspect first — file layouts vary between mirrors of these datasets,
    # and this prints exactly what was discovered and parsed before you commit.
    python har_datasets.py --dataset realworld --root /path/to/realworld --inspect

    # Build the windowed npz
    python har_datasets.py --dataset realworld --root /path/to/realworld --build

    # Then point the experiment at it
    python gait_dual_adversarial_v2.py --npz ./har_cache/realworld_hz50_w2.56.npz ...

If --inspect shows zero recordings or mangled labels, fix the regex for that
dataset (_REALWORLD_FILE_RE, _MOBIACT_FILE_RE, or the MotionSense folder pattern)
rather than running the pipeline on a partial parse. Silent partial parses are how a dataset
quietly becomes 3 subjects.

Requires: numpy, pandas
"""

import argparse
import glob
import json
import os
import re
import zipfile
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

GRAVITY = 9.80665

# =============================================================================
# Canonical vocabulary — so activity labels mean the same thing across datasets
# =============================================================================
CANONICAL_ACTIVITY = {
    # HHAR
    'bike': 'bike', 'sit': 'sit', 'stand': 'stand',
    'walk': 'walk', 'stairsup': 'stairsup', 'stairsdown': 'stairsdown',
    # RealWorld
    'walking': 'walk', 'running': 'jog', 'jogging': 'jog',
    'climbingup': 'stairsup', 'climbingdown': 'stairsdown',
    'sitting': 'sit', 'standing': 'stand', 'lying': 'lying', 'jumping': 'jump',
    # MotionSense folder codes
    'wlk': 'walk', 'jog': 'jog', 'ups': 'stairsup', 'dws': 'stairsdown',
    'sit_': 'sit', 'std': 'stand',
    # MobiAct activity codes (ADL subset; falls handled separately)
    'wal': 'walk', 'stu': 'stairsup', 'stn': 'stairsdown', 'jum': 'jump',
    'sch': 'sit_transition', 'chu': 'stand_transition',
    'csi': 'car_in', 'cso': 'car_out', 'lyi': 'lying',
}

# MobiAct fall events. Brief transients, not sustained activities — excluded by
# default. Pass them to --activities explicitly if you want them.
MOBIACT_FALLS = {'fol': 'fall_forward', 'fkl': 'fall_knees',
                 'bsc': 'fall_backsitting', 'sdl': 'fall_sideward'}

AMBULATORY = ['walk', 'jog', 'stairsup', 'stairsdown', 'bike']


def canon_activity(raw):
    key = str(raw).strip().lower()
    return CANONICAL_ACTIVITY.get(key, key)


# =============================================================================
# Config + containers
# =============================================================================
@dataclass
class WindowConfig:
    target_hz: float = 50.0
    window_seconds: float = 2.56
    overlap: float = 0.5
    max_gap_s: float = 1.0
    min_segment_s: float = 4.0

    def tag(self):
        return (f"hz{self.target_hz:g}_w{self.window_seconds:g}"
                f"_ov{self.overlap:g}_gap{self.max_gap_s:g}_min{self.min_segment_s:g}")


@dataclass
class Recording:
    """One contiguous (subject, activity, nuisance) capture, before windowing."""
    t_sec: np.ndarray          # (N,) seconds, monotonically increasing
    signal: np.ndarray         # (N, 6) acc xyz (m/s^2, gravity-inclusive) + gyro xyz (rad/s)
    subject: str
    activity: str
    nuisance: str
    source: str = ''           # file path, for the inspect report


@dataclass
class WindowedDataset:
    X: np.ndarray
    subject: np.ndarray
    activity: np.ndarray
    nuisance: np.ndarray
    seg_id: np.ndarray
    meta: dict = field(default_factory=dict)

    def summary(self):
        s = [f"dataset      : {self.meta.get('dataset')}",
             f"nuisance     : {self.meta.get('nuisance_name')} "
             f"({'usable' if self.meta.get('has_nuisance') else 'CONSTANT — skip the device head'})",
             f"windows      : {len(self.X):,}  shape {self.X.shape}",
             f"segments     : {len(np.unique(self.seg_id)):,}",
             f"subjects ({len(np.unique(self.subject))}) : {sorted(np.unique(self.subject))[:12]}"
             f"{' ...' if len(np.unique(self.subject)) > 12 else ''}",
             f"activities({len(np.unique(self.activity))}) : {sorted(np.unique(self.activity))}",
             f"nuisance  ({len(np.unique(self.nuisance))}) : {sorted(np.unique(self.nuisance))}"]
        n_sub = len(np.unique(self.subject))
        s.append(f"chance identity accuracy: {100.0 / max(n_sub, 1):.1f}%")
        return "\n".join("  " + line for line in s)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            # canonical names
            subject=self.subject, activity=self.activity, nuisance=self.nuisance,
            # aliases the experiment script already reads
            y_subject=self.subject, y_activity=self.activity, y_device=self.nuisance,
            seg_id=self.seg_id, meta=json.dumps(self.meta))
        return path

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        return WindowedDataset(d['X'], d['subject'], d['activity'], d['nuisance'],
                               d['seg_id'], json.loads(str(d['meta'])))


# =============================================================================
# Shared resampling / segmentation / windowing
# =============================================================================
def _segment_and_resample(t_sec, signal, cfg):
    order = np.argsort(t_sec)
    t_sec, signal = np.asarray(t_sec)[order], np.asarray(signal)[order]

    out = []
    if len(t_sec) < 8:
        return out
    gaps = np.where(np.diff(t_sec) > cfg.max_gap_s)[0]
    bounds = np.concatenate([[0], gaps + 1, [len(t_sec)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        ts, sg = t_sec[a:b], signal[a:b]
        if len(ts) < 8:
            continue
        dur = ts[-1] - ts[0]
        if dur < cfg.min_segment_s:
            continue
        n_new = int(dur * cfg.target_hz) + 1
        grid = ts[0] + np.arange(n_new) / cfg.target_hz
        res = np.column_stack([np.interp(grid, ts, sg[:, c]) for c in range(sg.shape[1])])
        out.append(res.astype(np.float32))
    return out


def window_recordings(records: List[Recording], cfg: WindowConfig, dataset: str,
                      nuisance_name: str):
    win_len = int(round(cfg.window_seconds * cfg.target_hz))
    step = max(1, int(round(win_len * (1.0 - cfg.overlap))))

    X, subj, act, nui, seg = [], [], [], [], []
    seg_counter = 0
    for rec in records:
        for chunk in _segment_and_resample(rec.t_sec, rec.signal, cfg):
            if len(chunk) < win_len:
                continue
            n_before = len(X)
            for s in range(0, len(chunk) - win_len + 1, step):
                X.append(chunk[s:s + win_len])
                subj.append(rec.subject)
                act.append(rec.activity)
                nui.append(rec.nuisance)
                seg.append(seg_counter)
            if len(X) > n_before:
                seg_counter += 1

    if not X:
        raise RuntimeError(
            f"{dataset}: windowing produced nothing. Either no recordings were parsed "
            f"(run --inspect) or every segment was shorter than "
            f"max(window_seconds={cfg.window_seconds}, min_segment_s={cfg.min_segment_s}).")

    nui_arr = np.asarray(nui)
    meta = dict(dataset=dataset, nuisance_name=nuisance_name,
                has_nuisance=bool(len(np.unique(nui_arr)) > 1),
                window_config=asdict(cfg), channels='ax,ay,az,gx,gy,gz',
                acc_units='m/s^2 (gravity-inclusive)', gyro_units='rad/s')
    return WindowedDataset(np.asarray(X, dtype=np.float32), np.asarray(subj),
                           np.asarray(act), nui_arr,
                           np.asarray(seg, dtype=np.int64), meta)


# =============================================================================
# Utilities
# =============================================================================
def _extract_zips(root, verbose=True):
    """RealWorld ships per-subject zips of CSVs. Extract each once, in place."""
    n = 0
    for zp in glob.glob(os.path.join(root, '**', '*.zip'), recursive=True):
        dest = os.path.splitext(zp)[0]
        if os.path.isdir(dest):
            continue
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
            n += 1
        except zipfile.BadZipFile:
            if verbose:
                print(f"  (skipping unreadable zip: {zp})")
    if n and verbose:
        print(f"  extracted {n} zip archive(s)")
    return n


def _median_rate(t_sec):
    if len(t_sec) < 10:
        return float('nan')
    dt = np.diff(np.sort(np.asarray(t_sec, dtype=np.float64)))
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else float('nan')


def _merge_acc_gyro(t_a, sig_a, t_g, sig_g, tolerance_s=0.02):
    """Nearest-timestamp merge of two 3-axis streams onto the accelerometer clock."""
    a = pd.DataFrame({'t': t_a, 'ax': sig_a[:, 0], 'ay': sig_a[:, 1], 'az': sig_a[:, 2]})
    g = pd.DataFrame({'t': t_g, 'gx': sig_g[:, 0], 'gy': sig_g[:, 1], 'gz': sig_g[:, 2]})
    m = pd.merge_asof(a.sort_values('t'), g.sort_values('t'), on='t',
                      direction='nearest', tolerance=float(tolerance_s)).dropna()
    if len(m) < 8:
        return None, None
    return (m['t'].to_numpy(),
            m[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].to_numpy(dtype=np.float32))


# =============================================================================
# HHAR
# =============================================================================
_HHAR_COLS = ['Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'Device', 'gt']
_HHAR_DTYPES = {'x': np.float32, 'y': np.float32, 'z': np.float32,
                'User': 'category', 'Model': 'category',
                'Device': 'category', 'gt': 'category'}


def _find(root, name):
    for r, _, files in os.walk(root):
        if '__MACOSX' in r:
            continue
        if name in files:
            return os.path.join(r, name)
    return None


def load_hhar(root, cfg, exclude_devices=None, verbose=True) -> List[Recording]:
    acc_p = _find(root, 'Phones_accelerometer.csv')
    gyr_p = _find(root, 'Phones_gyroscope.csv')
    if not (acc_p and gyr_p):
        raise FileNotFoundError(f"HHAR phone CSVs not found under {root}")

    if verbose:
        print("  reading HHAR phone CSVs (large — this takes a minute)...")
    df_a = pd.read_csv(acc_p, usecols=_HHAR_COLS, dtype=_HHAR_DTYPES)
    df_g = pd.read_csv(gyr_p, usecols=_HHAR_COLS, dtype=_HHAR_DTYPES)
    for d in (df_a, df_g):
        d.drop(d.index[d['gt'].isna() | (d['gt'].astype(str) == 'null')], inplace=True)

    if exclude_devices:
        drop = set(exclude_devices)
        df_a = df_a[~(df_a['Model'].astype(str).isin(drop) | df_a['Device'].astype(str).isin(drop))]
        df_g = df_g[~(df_g['Model'].astype(str).isin(drop) | df_g['Device'].astype(str).isin(drop))]

    records = []
    keys = df_a[['User', 'Device', 'gt']].drop_duplicates().itertuples(index=False)
    for user, dev, act in keys:
        ga = df_a[(df_a['User'] == user) & (df_a['Device'] == dev) & (df_a['gt'] == act)]
        gg = df_g[(df_g['User'] == user) & (df_g['Device'] == dev) & (df_g['gt'] == act)]
        if len(ga) < 50 or len(gg) < 50:
            continue

        ta = ga['Creation_Time'].to_numpy(dtype=np.float64) / 1e9
        tg = gg['Creation_Time'].to_numpy(dtype=np.float64) / 1e9
        if not (5 < _median_rate(ta) < 1000):                     # documented HHAR quirk
            ta = ga['Arrival_Time'].to_numpy(dtype=np.float64) / 1e3
            tg = gg['Arrival_Time'].to_numpy(dtype=np.float64) / 1e3

        t, sig = _merge_acc_gyro(ta - ta.min(), ga[['x', 'y', 'z']].to_numpy(),
                                 tg - ta.min(), gg[['x', 'y', 'z']].to_numpy())
        if t is None:
            continue
        records.append(Recording(t, sig, str(user), canon_activity(act),
                                 str(ga['Model'].iloc[0]), source=str(dev)))
    return records


# =============================================================================
# RealWorld (HAR) 2016
# =============================================================================
_REALWORLD_POSITIONS = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']

# Filenames in the wild look like:
#   acc_walking_chest.csv        Gyroscope_walking_chest.csv
#   acc_climbingdown_2_chest.csv Gyroscope_climbingdown_2_chest.csv
# If --inspect reports 0 parsed files, widen this regex to match your copy.
_REALWORLD_FILE_RE = re.compile(
    r'^(?P<sensor>acc|accelerometer|gyr|gyroscope)'
    r'_(?P<activity>[a-z]+)'
    r'(?:_(?P<session>\d+))?'
    r'_(?P<position>' + '|'.join(_REALWORLD_POSITIONS) + r')'
    r'\.csv$', re.IGNORECASE)

_REALWORLD_SUBJECT_RE = re.compile(r'proband\s*(\d+)', re.IGNORECASE)


def _realworld_scan(root):
    """Return a list of parsed file descriptors, plus the paths that did not parse."""
    hits, misses = [], []
    for path in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True):
        if '__MACOSX' in path:
            continue
        m = _REALWORLD_FILE_RE.match(os.path.basename(path))
        if not m:
            misses.append(path)
            continue
        sm = _REALWORLD_SUBJECT_RE.search(path)
        if not sm:
            misses.append(path)
            continue
        sensor = m.group('sensor').lower()
        hits.append(dict(path=path,
                         subject=f"proband{int(sm.group(1))}",
                         sensor='acc' if sensor.startswith('acc') else 'gyro',
                         activity=canon_activity(m.group('activity')),
                         session=m.group('session') or '1',
                         position=m.group('position').lower()))
    return hits, misses


def _read_realworld_csv(path):
    """RealWorld CSVs: id, attr_time (unix ms), attr_x, attr_y, attr_z."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get('attr_time') or cols.get('time') or cols.get('timestamp')
    xs = [cols.get(f'attr_{a}') or cols.get(a) for a in ('x', 'y', 'z')]
    if tcol is None or any(c is None for c in xs):
        raise ValueError(f"unexpected columns in {path}: {list(df.columns)}")
    t = df[tcol].to_numpy(dtype=np.float64) / 1e3                 # ms -> s
    sig = df[xs].to_numpy(dtype=np.float32)
    return t, sig


def load_realworld(root, cfg, verbose=True) -> List[Recording]:
    _extract_zips(root, verbose)
    hits, misses = _realworld_scan(root)
    if not hits:
        raise FileNotFoundError(
            f"No RealWorld CSVs parsed under {root}. Run --inspect and widen "
            f"_REALWORLD_FILE_RE to match your copy's filenames.")

    by_key = {}
    for h in hits:
        by_key.setdefault((h['subject'], h['activity'], h['session'], h['position']),
                          {})[h['sensor']] = h['path']

    records = []
    for (subject, activity, _sess, position), pair in sorted(by_key.items()):
        if 'acc' not in pair or 'gyro' not in pair:
            continue                                              # need both streams
        try:
            ta, sa = _read_realworld_csv(pair['acc'])
            tg, sg = _read_realworld_csv(pair['gyro'])
        except Exception as e:                                    # noqa: BLE001
            if verbose:
                print(f"  (skipping {pair['acc']}: {e})")
            continue
        t0 = ta.min()
        t, sig = _merge_acc_gyro(ta - t0, sa, tg - t0, sg)
        if t is None:
            continue
        records.append(Recording(t, sig, subject, activity, position,
                                 source=os.path.basename(pair['acc'])))
    return records


# =============================================================================
# MotionSense
# =============================================================================
_MOTIONSENSE_DIR_RE = re.compile(r'^(dws|ups|wlk|jog|sit|std)_(\d+)$', re.IGNORECASE)
_MOTIONSENSE_SUB_RE = re.compile(r'^sub_(\d+)\.csv$', re.IGNORECASE)
_MOTIONSENSE_ACT = {'dws': 'stairsdown', 'ups': 'stairsup', 'wlk': 'walk',
                    'jog': 'jog', 'sit': 'sit', 'std': 'stand'}
MOTIONSENSE_NUISANCE = 'iphone6s_frontpocket'
MOTIONSENSE_HZ = 50.0          # DeviceMotion is sampled at 50 Hz; rows carry no clock


def _motionsense_scan(root):
    hits, misses = [], []
    for path in glob.glob(os.path.join(root, '**', 'sub_*.csv'), recursive=True):
        if '__MACOSX' in path:
            continue
        sm = _MOTIONSENSE_SUB_RE.match(os.path.basename(path))
        dm = _MOTIONSENSE_DIR_RE.match(os.path.basename(os.path.dirname(path)))
        if not (sm and dm):
            misses.append(path)
            continue
        hits.append(dict(path=path, subject=f"sub{int(sm.group(1))}",
                         activity=_MOTIONSENSE_ACT[dm.group(1).lower()],
                         trial=dm.group(2)))
    return hits, misses


def load_motionsense(root, cfg, verbose=True) -> List[Recording]:
    hits, _ = _motionsense_scan(root)
    if not hits:
        raise FileNotFoundError(
            f"No MotionSense sub_*.csv found under {root}. Expected the "
            f"A_DeviceMotion_data layout: <act>_<trial>/sub_<k>.csv")

    records = []
    for h in sorted(hits, key=lambda d: (d['subject'], d['activity'], d['trial'])):
        df = pd.read_csv(h['path'])
        cols = {c.lower().strip(): c for c in df.columns}

        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        ua = [col(f'useracceleration.{a}', f'userAcceleration.{a}'.lower()) for a in 'xyz']
        gr = [col(f'gravity.{a}') for a in 'xyz']
        rr = [col(f'rotationrate.{a}') for a in 'xyz']
        if any(c is None for c in ua + gr + rr):
            if verbose:
                print(f"  (skipping {h['path']}: columns {list(df.columns)[:8]}...)")
            continue

        # MotionSense stores userAcceleration with gravity REMOVED, both in g.
        # HHAR/RealWorld accelerometers are gravity-inclusive m/s^2 — add gravity
        # back and convert, so the channel means the same thing in every dataset.
        acc = (df[ua].to_numpy(dtype=np.float32) + df[gr].to_numpy(dtype=np.float32)) * GRAVITY
        gyro = df[rr].to_numpy(dtype=np.float32)
        sig = np.concatenate([acc, gyro], axis=1)
        t = np.arange(len(sig), dtype=np.float64) / MOTIONSENSE_HZ   # no clock column

        records.append(Recording(t, sig, h['subject'], h['activity'],
                                 MOTIONSENSE_NUISANCE, source=os.path.basename(h['path'])))
    return records


# =============================================================================
# MobiAct v2.0
# =============================================================================
# Files are named <ACT>_<subject>_<trial>_annotated.csv, e.g. WAL_1_1_annotated.csv,
# and unlike HHAR/RealWorld they carry accelerometer AND gyroscope in one file, so
# no cross-sensor merge is needed.
_MOBIACT_FILE_RE = re.compile(
    r'^(?P<act>[A-Za-z]{3})_(?P<subj>\d+)_(?P<trial>\d+)(?:_annotated)?\.csv$')

MOBIACT_NUISANCE = 'galaxys3_trouserpocket'


def _mobiact_scan(root, include_falls=False):
    hits, misses = [], []
    for path in glob.glob(os.path.join(root, '**', '*.csv'), recursive=True):
        if '__MACOSX' in path:
            continue
        m = _MOBIACT_FILE_RE.match(os.path.basename(path))
        if not m:
            misses.append(path)
            continue
        code = m.group('act').lower()
        if code in MOBIACT_FALLS and not include_falls:
            continue
        activity = MOBIACT_FALLS[code] if code in MOBIACT_FALLS else canon_activity(code)
        hits.append(dict(path=path, subject=f"sub{int(m.group('subj'))}",
                         activity=activity, trial=m.group('trial'), code=code))
    return hits, misses


def load_mobiact(root, cfg, include_falls=False, verbose=True) -> List[Recording]:
    hits, _ = _mobiact_scan(root, include_falls)
    if not hits:
        raise FileNotFoundError(
            f"No MobiAct CSVs found under {root}. Expected files named like "
            f"'WAL_1_1_annotated.csv' (Annotated Data folder).")

    records, skipped = [], 0
    for h in sorted(hits, key=lambda d: (d['subject'], d['activity'], d['trial'])):
        try:
            df = pd.read_csv(h['path'])
        except Exception:                                          # noqa: BLE001
            skipped += 1
            continue
        cols = {c.lower().strip(): c for c in df.columns}

        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        acc = [col(f'acc_{a}', f'ax' if a == 'x' else f'a{a}') for a in 'xyz']
        gyr = [col(f'gyro_{a}', f'gyr_{a}', f'g{a}') for a in 'xyz']
        if any(c is None for c in acc + gyr):
            if verbose and skipped < 3:
                print(f"  (skipping {os.path.basename(h['path'])}: columns "
                      f"{list(df.columns)[:8]})")
            skipped += 1
            continue

        tcol = col('rel_time', 'reltime')
        if tcol is not None:
            t = df[tcol].to_numpy(dtype=np.float64)                # already seconds
        else:
            tcol = col('timestamp', 'time')
            if tcol is None:
                skipped += 1
                continue
            t = df[tcol].to_numpy(dtype=np.float64) / 1e9          # ns -> s
        t = t - np.nanmin(t)

        sig = np.concatenate([df[acc].to_numpy(dtype=np.float32),
                              df[gyr].to_numpy(dtype=np.float32)], axis=1)
        ok = np.isfinite(t) & np.isfinite(sig).all(axis=1)
        if ok.sum() < 32:
            skipped += 1
            continue

        records.append(Recording(t[ok], sig[ok], h['subject'], h['activity'],
                                 MOBIACT_NUISANCE, source=os.path.basename(h['path'])))

    if verbose and skipped:
        print(f"  skipped {skipped} unreadable/short file(s)")
    return records


# =============================================================================
# Dispatch
# =============================================================================
LOADERS = {'hhar': load_hhar, 'realworld': load_realworld,
           'motionsense': load_motionsense, 'mobiact': load_mobiact}
NUISANCE_NAME = {'hhar': 'device', 'realworld': 'position',
                 'motionsense': 'none', 'mobiact': 'none'}


def build(dataset, root, cfg, exclude_devices=None, activities=None,
          include_falls=False, verbose=True):
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset '{dataset}' (have: {sorted(LOADERS)})")

    if verbose:
        print(f"\nLoading {dataset} from {root}")
    kw = dict(exclude_devices=exclude_devices) if dataset == 'hhar' else {}
    if dataset == 'mobiact':
        kw = dict(include_falls=include_falls)
    records = LOADERS[dataset](root, cfg, verbose=verbose, **kw)
    if verbose:
        print(f"  parsed {len(records)} recordings")

    if activities:
        keep = {canon_activity(a) for a in activities}
        before = len(records)
        records = [r for r in records if r.activity in keep]
        if verbose:
            print(f"  activity filter {sorted(keep)}: kept {len(records)}/{before} recordings")
        if not records:
            raise ValueError(f"activity filter removed everything; present activities "
                             f"differ from {sorted(keep)} — run --inspect")

    ds = window_recordings(records, cfg, dataset, NUISANCE_NAME[dataset])
    return ds


def inspect(dataset, root, cfg):
    print(f"\n{'='*78}\nINSPECT — {dataset}\n{'='*78}")

    if dataset == 'realworld':
        _extract_zips(root)
        hits, misses = _realworld_scan(root)
    elif dataset == 'motionsense':
        hits, misses = _motionsense_scan(root)
    elif dataset == 'mobiact':
        hits, misses = _mobiact_scan(root)
    else:
        hits, misses = [], []
        acc = _find(root, 'Phones_accelerometer.csv')
        print(f"  Phones_accelerometer.csv: {acc or 'NOT FOUND'}")
        print(f"  Phones_gyroscope.csv    : {_find(root, 'Phones_gyroscope.csv') or 'NOT FOUND'}")

    if hits:
        df = pd.DataFrame(hits)
        print(f"  parsed files: {len(df)}")
        for c in ('subject', 'activity', 'position', 'sensor', 'trial', 'code'):
            if c in df.columns:
                vals = sorted(df[c].astype(str).unique())
                print(f"  {c:<9}({len(vals):>3}): {vals[:14]}{' ...' if len(vals) > 14 else ''}")
        print("\n  sample paths:")
        for p in df['path'].head(3):
            print(f"    {p}")

    if misses:
        print(f"\n  {len(misses)} CSV(s) did NOT parse. First few:")
        for p in misses[:8]:
            print(f"    {p}")
        print("  If these look like real data files, the filename regex needs widening —\n"
              "  a partial parse silently shrinks your subject count.")

    print("\n  Building a small sample to verify signal parsing...")
    ds = build(dataset, root, cfg, verbose=False)
    print(ds.summary())
    print(f"\n  signal check — per-channel mean/std over 2000 random windows:")
    idx = np.random.RandomState(0).choice(len(ds.X), min(2000, len(ds.X)), replace=False)
    flat = ds.X[idx].reshape(-1, ds.X.shape[-1])
    for i, name in enumerate(['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
        print(f"    {name}: mean {flat[:, i].mean():+8.3f}  std {flat[:, i].std():7.3f}")
    print("  (accelerometer is gravity-inclusive m/s^2, so one axis should sit near ±9.8)")


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Multi-dataset IMU adapter",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', required=True, choices=sorted(LOADERS))
    p.add_argument('--root', required=True, help="Folder containing the extracted dataset.")
    p.add_argument('--out_dir', default='./har_cache')
    p.add_argument('--inspect', action='store_true',
                   help="Report what was discovered and parsed, then stop. RUN THIS FIRST.")
    p.add_argument('--build', action='store_true', help="Write the windowed npz.")
    p.add_argument('--target_hz', type=float, default=50.0)
    p.add_argument('--window_seconds', type=float, default=2.56)
    p.add_argument('--overlap', type=float, default=0.5)
    p.add_argument('--max_gap_s', type=float, default=1.0)
    p.add_argument('--min_segment_s', type=float, default=4.0)
    p.add_argument('--activities', nargs='+', default=None,
                   help="Restrict to these (canonical names). Ambulatory set: "
                        + ' '.join(AMBULATORY))
    p.add_argument('--exclude_devices', nargs='+', default=None, help="HHAR only.")
    p.add_argument('--include_falls', action='store_true',
                   help="MobiAct only: keep the four fall events. They are brief transients "
                        "rather than sustained activities, so they are excluded by default.")
    args = p.parse_args()

    cfg = WindowConfig(args.target_hz, args.window_seconds, args.overlap,
                       args.max_gap_s, args.min_segment_s)

    if args.inspect or not args.build:
        inspect(args.dataset, args.root, cfg)
        if not args.build:
            return

    ds = build(args.dataset, args.root, cfg, exclude_devices=args.exclude_devices,
               activities=args.activities, include_falls=args.include_falls)
    print("\n" + ds.summary())

    act_tag = 'all' if not args.activities else '-'.join(sorted(canon_activity(a)
                                                                for a in args.activities))
    ex_tag = '' if not args.exclude_devices else '_ex-' + '-'.join(sorted(args.exclude_devices))
    path = os.path.join(args.out_dir,
                        f"{args.dataset}_{cfg.tag()}_acts-{act_tag}{ex_tag}.npz")
    ds.save(path)
    print(f"\nSaved -> {path}")
    if not ds.meta['has_nuisance']:
        print(f"  NOTE: {args.dataset} has a single {NUISANCE_NAME[args.dataset]} level. "
              f"Run it WITHOUT the nuisance-adversarial branch.")


if __name__ == '__main__':
    main()

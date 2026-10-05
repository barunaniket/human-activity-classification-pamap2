"""PAMAP2 loading, cleaning, row-level and windowed feature construction."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "PAMAP2_Dataset" / "Protocol"
CACHE = ROOT / "data" / "pamap2_clean.parquet"

# The 12 protocol activities used in the paper
ACTIVITIES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic walking", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    24: "rope jumping",
}
LABEL_IDS = sorted(ACTIVITIES)
ID_TO_IDX = {a: i for i, a in enumerate(LABEL_IDS)}
CLASS_NAMES = [ACTIVITIES[a] for a in LABEL_IDS]

# Column layout of the .dat files: timestamp, activityID, heart rate, then 3 IMUs x 17 columns
_IMU_COLS = ["temp", "acc16_x", "acc16_y", "acc16_z", "acc6_x", "acc6_y", "acc6_z",
             "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z",
             "ori_0", "ori_1", "ori_2", "ori_3"]
COLUMNS = (["timestamp", "activity", "heart_rate"]
           + [f"{imu}_{c}" for imu in ("hand", "chest", "ankle") for c in _IMU_COLS])

FULL_FEATURES = [c for c in COLUMNS if c not in ("timestamp", "activity")]  # 52, as in the paper
LIMITED_FEATURES = ["heart_rate"] + [c for c in FULL_FEATURES if c.startswith("hand_")]  # smartwatch-like
FEATURE_SETS = {"full": FULL_FEATURES, "limited": LIMITED_FEATURES}

FS = 100  # IMU sampling rate (Hz)


def load_raw() -> pd.DataFrame:
    """Read the 9 subject files, keep the 12 activities, interpolate heart rate."""
    frames = []
    for f in sorted(RAW_DIR.glob("subject*.dat")):
        df = pd.read_csv(f, sep=r"\s+", header=None, names=COLUMNS, na_values="NaN")
        df["subject"] = int(f.stem.replace("subject", ""))
        # heart rate is ~9 Hz vs 100 Hz IMUs: fill by linear interpolation (as in the paper)
        df["heart_rate"] = df["heart_rate"].interpolate(limit_direction="both")
        df = df[df["activity"].isin(LABEL_IDS)]
        # IMU dropouts: interpolate short gaps, then drop what is left
        df[FULL_FEATURES] = df[FULL_FEATURES].interpolate(limit_direction="both")
        df = df.dropna(subset=FULL_FEATURES)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["y"] = out["activity"].map(ID_TO_IDX).astype(np.int64)
    return out


def load_clean(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)
    df = load_raw()
    try:
        df.to_parquet(CACHE)
    except ImportError:  # parquet engine not installed -> just skip caching
        pass
    return df


def row_level(df: pd.DataFrame, features, step: int = 10):
    """Per-timestep samples (paper setting). `step` decimates 100 Hz -> 100/step Hz."""
    parts = [g.iloc[::step] for _, g in df.groupby("subject", sort=True)]
    d = pd.concat(parts)
    return d[features].to_numpy(np.float32), d["y"].to_numpy(), d["subject"].to_numpy()


def make_windows(df: pd.DataFrame, features, win_s: float = 2.0, overlap: float = 0.5,
                 purity: float = 0.9):
    """Sliding windows over each subject's recording -> raw array (N, T, C).

    Windows never cross a label change; a window is kept only if one class covers
    >= `purity` of its samples, and gets that majority label.
    """
    win = int(win_s * FS)
    stride = max(1, int(win * (1 - overlap)))
    X, y, subj = [], [], []
    for s, g in df.groupby("subject", sort=True):
        vals = g[features].to_numpy(np.float32)
        lab = g["y"].to_numpy()
        for start in range(0, len(g) - win + 1, stride):
            seg_lab = lab[start:start + win]
            maj = np.bincount(seg_lab, minlength=len(LABEL_IDS)).argmax()
            if (seg_lab == maj).mean() >= purity:
                X.append(vals[start:start + win])
                y.append(maj)
                subj.append(s)
    return np.stack(X), np.array(y), np.array(subj)


def window_stats(Xw: np.ndarray) -> np.ndarray:
    """(N, T, C) -> (N, 5*C) hand-crafted features: mean, std, min, max, mean |diff|."""
    return np.concatenate([
        Xw.mean(1), Xw.std(1), Xw.min(1), Xw.max(1), np.abs(np.diff(Xw, axis=1)).mean(1),
    ], axis=1).astype(np.float32)

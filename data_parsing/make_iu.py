"""
IU dataset parser (custom).

Expected columns (CSV):
activity, time_s,
lw_x, lw_y, lw_z,
lh_x, lh_y, lh_z,
la_x, la_y, la_z,
ra_x, ra_y, ra_z

We will use triaxial acceleration from: lw_x, lw_y, lw_z

Input files:
  iu_data/id*.csv

Assumptions:
- Sample rate is 100 Hz (so 10 seconds = 1000 rows).
- time_s is numeric seconds (can be float). If time_s is missing or not usable,
  we fall back to row indices / sample counts.
- If activity is present, we store it as Y (string label per window) using the
  first row's value inside the window. If activity varies inside the window,
  we can optionally reject that window (default: reject).
"""

import glob
import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEVICE_HZ = 100  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds (match capture24_100hz_w10_o0 style)

WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN

# tolerance for time_s-based window duration check (1% like other scripts)
WINDOW_TOL = 0.01

DATAFILES = "iu_data/id*.csv"
OUTDIR = "iu_100hz_w10_o0/"

# If True: reject windows where activity changes within the window
REQUIRE_CONSTANT_ACTIVITY = True


def parse_pid_from_path(path: str) -> str:
    """
    Extract an ID from filename. For iu_data/id*.csv, we return the matched "idXXX"
    stem (without extension). You can modify this if your naming differs.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    # try to find something like id123 or id_123
    m = re.search(r"(id[_-]?\w+)", stem, flags=re.IGNORECASE)
    return m.group(1) if m else stem


def is_good_quality_window(df_window: pd.DataFrame) -> bool:
    if len(df_window) != WINDOW_LEN:
        return False

    # Any NaNs in required accel columns => reject
    if df_window[["lw_x", "lw_y", "lw_z"]].isna().any().any():
        return False

    # Optional: activity must be constant
    if REQUIRE_CONSTANT_ACTIVITY and "activity" in df_window.columns:
        # treat NaN as its own category; if all NaN, we accept as unlabeled
        uniq = pd.Series(df_window["activity"]).dropna().unique()
        if len(uniq) > 1:
            return False

    # If time_s exists, do a duration sanity check similar to other parsers
    if "time_s" in df_window.columns:
        try:
            t0 = float(df_window["time_s"].iloc[0])
            t1 = float(df_window["time_s"].iloc[-1])
            w_duration = t1 - t0
            target_duration = float(WINDOW_SEC)
            if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
                # Depending on your data, this might be too strict.
                return False
        except Exception:
            # If time_s is not parseable, don't fail the window for that reason.
            pass

    return True


def main():
    X, Y, T, P = [], [], [], []

    files = sorted(glob.glob(DATAFILES))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No files matched DATAFILES='{DATAFILES}'. "
            f"Check your working directory and path."
        )

    for datafile in tqdm(files, desc="Parsing IU CSVs"):
        df = pd.read_csv(datafile)

        # Basic column checks
        required = ["lw_x", "lw_y", "lw_z"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {datafile}")

        pid = parse_pid_from_path(datafile)

        # Optional: if there are multiple activities per file, we still handle it window-by-window
        for i in range(0, len(df), WINDOW_STEP_LEN):
            w = df.iloc[i : i + WINDOW_LEN]

            if not is_good_quality_window(w):
                continue

            x = w[["lw_x", "lw_y", "lw_z"]].to_numpy(dtype=np.float32)

            # window "time": prefer time_s, else use starting sample index converted to seconds
            if "time_s" in w.columns:
                try:
                    t = float(w["time_s"].iloc[0])
                except Exception:
                    t = float(i) / DEVICE_HZ
            else:
                t = float(i) / DEVICE_HZ

            # label: if activity column exists and not all NaN, use it; else "unknown"
            if "activity" in w.columns:
                y0 = w["activity"].iloc[0]
                y = "unknown" if pd.isna(y0) else str(y0)
            else:
                y = "unknown"

            X.append(x)
            Y.append(y)
            T.append(t)
            P.append(pid)

    X = np.asarray(X, dtype=np.float32)  # (N, 1000, 3)
    Y = np.asarray(Y)                    # (N,)
    T = np.asarray(T, dtype=np.float64)  # (N,)
    P = np.asarray(P)                    # (N,)

    os.makedirs(OUTDIR, exist_ok=True)
    np.save(os.path.join(OUTDIR, "X.npy"), X)
    np.save(os.path.join(OUTDIR, "Y.npy"), Y)
    np.save(os.path.join(OUTDIR, "time.npy"), T)
    np.save(os.path.join(OUTDIR, "pid.npy"), P)

    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:")
    print(pd.Series(Y).value_counts().head(20))
    print("Num subjects:", len(np.unique(P)))


if __name__ == "__main__":
    main()
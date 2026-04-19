"""
One-off data audit: scans every parquet file in data/ for illegal values.

Rules applied per column (based on physical meaning documented in CLAUDE.md):
  * s4 / start_weight / end_weight / weight gains: must be >= 0 (mass can't be negative)
  * dt (daily transpiration, grams/day): must be >= 0
  * wstemp (air temp, C): sanity window [-10, 60]
  * wsrh (relative humidity, %): must be in [0, 100]
  * soil_sand: must be in {0, 1}
  * day_num: must be >= 0
  * timestamp: must not be NaT, must be monotonic per unique_id

Also reports NaN / inf counts and exact duplicates.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Expected non-negative columns (any numeric weight-like or rate column)
NON_NEGATIVE_COLS = {
    "s4", "start_weight", "end_weight", "dt", "day_num",
    "avg_temp_light", "avg_humidity_light",  # if these exist anywhere
}
# Columns with a plausible physical range
RANGE_COLS = {
    "wstemp":       (-10.0, 60.0),   # C
    "wsrh":         (0.0, 100.0),    # %
    "avg_temp":     (-10.0, 60.0),
    "avg_humidity": (0.0, 100.0),
}
BINARY_COLS = {"soil_sand"}


def audit_file(path: str) -> int:
    print("=" * 90)
    print(f"FILE: {os.path.relpath(path, DATA_DIR)}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  ERROR reading: {e}")
        return 1

    n = len(df)
    print(f"  rows={n}  cols={len(df.columns)}")
    issue_count = 0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. NaN / inf report
    for c in numeric_cols:
        nan_ct = int(df[c].isna().sum())
        inf_ct = int(np.isinf(df[c].to_numpy(dtype=float, na_value=0.0)).sum())
        if nan_ct or inf_ct:
            print(f"  [{c}] NaN={nan_ct}  inf={inf_ct}")
            if inf_ct:
                issue_count += inf_ct

    # 2. Negative values where non-negative expected
    for c in numeric_cols:
        if c in NON_NEGATIVE_COLS or c.endswith("_weight") or "weight" in c:
            neg = df[df[c] < 0]
            if len(neg):
                print(f"  [{c}] NEGATIVE values: {len(neg)} rows  "
                      f"min={neg[c].min():.4f}  max={neg[c].max():.4f}")
                issue_count += len(neg)

    # 3. Out-of-range values
    for c, (lo, hi) in RANGE_COLS.items():
        if c not in df.columns:
            continue
        bad = df[(df[c].notna()) & ((df[c] < lo) | (df[c] > hi))]
        if len(bad):
            print(f"  [{c}] OUT OF RANGE [{lo},{hi}]: {len(bad)} rows  "
                  f"min={bad[c].min():.4f}  max={bad[c].max():.4f}")
            issue_count += len(bad)

    # 4. Binary columns
    for c in BINARY_COLS:
        if c not in df.columns:
            continue
        unique_vals = set(pd.unique(df[c].dropna()).tolist())
        unexpected = unique_vals - {0, 1, 0.0, 1.0, True, False}
        if unexpected:
            print(f"  [{c}] UNEXPECTED values: {unexpected}")
            issue_count += 1

    # 5. Timestamp sanity per unique_id
    if "timestamp" in df.columns and "unique_id" in df.columns:
        nat = int(df["timestamp"].isna().sum())
        if nat:
            print(f"  [timestamp] NaT count: {nat}")
            issue_count += nat
        non_monotonic = 0
        for uid, grp in df.groupby("unique_id", sort=False):
            ts = grp["timestamp"].dropna()
            if not ts.is_monotonic_increasing:
                non_monotonic += 1
        if non_monotonic:
            print(f"  [timestamp] non-monotonic plants: {non_monotonic}")
            # not necessarily an error (depends on pipeline), so no count bump

    # 6. end_weight < start_weight on the same day -> negative weight gain
    if {"start_weight", "end_weight"}.issubset(df.columns):
        loss = df[df["end_weight"] < df["start_weight"]]
        if len(loss):
            diffs = (loss["end_weight"] - loss["start_weight"])
            print(f"  [weight] end<start (daily weight loss): {len(loss)} rows  "
                  f"min_delta={diffs.min():.4f}  median_delta={diffs.median():.4f}")
            # This is physically possible (pruning, water loss) — report but don't
            # treat as a hard error.

    # 7. Exact duplicate rows
    dups = int(df.duplicated().sum())
    if dups:
        print(f"  duplicate rows: {dups}")
        issue_count += dups

    # 8. Duplicate (unique_id, timestamp) pairs
    if {"unique_id", "timestamp"}.issubset(df.columns):
        key_dups = int(df.duplicated(subset=["unique_id", "timestamp"]).sum())
        if key_dups:
            print(f"  duplicate (unique_id,timestamp): {key_dups}")
            issue_count += key_dups

    # 9. Duplicate (unique_id, day_num) pairs for daily-summary files
    if {"unique_id", "day_num"}.issubset(df.columns):
        key_dups = int(df.duplicated(subset=["unique_id", "day_num"]).sum())
        if key_dups:
            print(f"  duplicate (unique_id,day_num): {key_dups}")
            issue_count += key_dups

    if issue_count == 0:
        print("  OK — no illegal values found.")
    else:
        print(f"  >>> {issue_count} issues flagged")
    return issue_count


def main() -> int:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    total = 0
    for fp in files:
        total += audit_file(fp)
    print("=" * 90)
    print(f"TOTAL issues across {len(files)} files: {total}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

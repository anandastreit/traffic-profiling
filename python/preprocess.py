"""
Preprocessing pipeline for PARAFAC traffic model.

Ports SUESTE_OLD preprocessing (mix of Python 2 + R) to a single Python 3 script.

Stages (run in order):
  1  extract  raw measurements → per-direction traffic CSVs       [stub — TBD]
  2  align    raw CSVs → daily 1440-minute series per user
  3  stats    daily series → NaN statistics (info only)
  4  filter   daily series → filtered weekly CSVs for train_model.py
  all          runs stages 2 + 3 + 4

Usage:
  python preprocess.py align   --measure down
  python preprocess.py filter  --measure down
  python preprocess.py stats   --measure down
  python preprocess.py all     --measure both

Data directory layout:
  data/raw/{down|up}/                 <- raw CSVs (input to Stage 2)
  data/daily/{down|up}/allSeries/     <- daily series (output of Stage 2)
  data/daily/{down|up}/10min/         <- strict daily series (max 10-min hole)
  data/daily/{down|up}/selected_allSeries.csv  <- optional week/flag selection file
  data/input/{down|up}/               <- filtered weekly CSVs (input to train_model.py)

Raw CSV expected columns: hostid (or mac) | timestamp (epoch or datetime) | traffic (bytes/sec)
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
DATA_DIR = _HERE / "data"

RAW_DIR = DATA_DIR / "raw"
DAILY_DIR = DATA_DIR / "daily"
INPUT_DIR = DATA_DIR / "input"

GRANULARITY = 60   # seconds per bucket (1 minute)
BYTE = 8           # conversion factor: bytes → bits

MAX_CONSEC_NAN = 180   # max allowed consecutive missing minutes (3 hours)
MAX_TOTAL_NAN = 720    # max allowed total missing minutes (50% of day)
NUM_MINUTES = 1440

_MINUTE_COLS = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(60)]
_COL_NAMES = ["user", "day"] + _MINUTE_COLS
_DAILY_PREFIX = "series_giga_"
_10MIN_PREFIX = "series_giga_10_"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _compute_max_consec_nan(series: pd.Series) -> int:
    """Return the length of the longest consecutive NaN run."""
    null_mask = series.isnull().astype(int)
    if null_mask.sum() == 0:
        return 0
    # cumsum of non-NaN values creates a group id per run; summing NaN flags per group gives run length
    return int(null_mask.groupby(series.notnull().astype(int).cumsum()).sum().max())


# ---------------------------------------------------------------------------
# Stage 1: extract (stub)
# ---------------------------------------------------------------------------

def extract(measure: str):
    """
    Stage 1: extract per-device traffic CSVs from raw measurement files.

    NOT IMPLEMENTED — data collection format is TBD for new collections.
    Port the logic from SUESTE_OLD/gigalink-processed/get_df.py once the new
    query/collection format is defined.

    Expected output in data/raw/{measure}/:
      downstream_traffic_*.csv  (measure=down)
      upstream_traffic_*.csv    (measure=up)
    Columns: hostid | timestamp | traffic (bytes/sec)
    """
    raise NotImplementedError(
        "Stage 1 (extract) is a stub. "
        "Implement once the new data collection format is known. "
        f"Expected output directory: {RAW_DIR / measure}"
    )


# ---------------------------------------------------------------------------
# Stage 2: align — port of make_UDs_traffic.R
# ---------------------------------------------------------------------------

def _round_to_minute(ts: datetime) -> datetime:
    discard = timedelta(seconds=ts.second, microseconds=ts.microsecond)
    base = ts - discard
    return base if discard < timedelta(seconds=30) else base + timedelta(minutes=1)


def _align_one_file(raw_csv: Path, out_daily: Path, out_10min: Path):
    """
    Align one raw traffic CSV to 1440-minute daily series per user.

    Three non-obvious behaviors ported from make_UDs_traffic.R:
    - Post-gap NaN: the minute after each missing minute is also set NaN —
      a measurement following a gap doesn't represent exactly one minute of traffic.
    - DST handling: duplicate timestamps (clock rollback) keep only the first occurrence.
    - Midnight carryover: a measurement rounded past 23:59 is attributed to
      minute[0] of the next day.
    """
    out_daily.mkdir(parents=True, exist_ok=True)
    out_10min.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_csv)
    raw.columns = [c.strip() for c in raw.columns]
    if "mac" in raw.columns and "hostid" not in raw.columns:
        raw = raw.rename(columns={"mac": "hostid"})

    # Try Unix epoch first, fall back to datetime string
    ts_parsed = pd.to_datetime(raw["timestamp"], unit="s", errors="coerce")
    if ts_parsed.isna().all():
        ts_parsed = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw["ts"] = ts_parsed
    raw = raw.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    raw["date"] = raw["ts"].dt.date.astype(str)

    last_day_user: dict = {}  # {hostid: bits_per_min} midnight carryover

    for current_day in sorted(raw["date"].unique()):
        day_df = raw[raw["date"] == current_day]
        users = day_df["hostid"].unique()

        minute_index = pd.date_range(
            start=f"{current_day} 00:00:00", periods=NUM_MINUTES, freq="1min"
        )
        slot_to_idx = {dt.strftime("%Y-%m-%d %H:%M"): i for i, dt in enumerate(minute_index)}

        rows_all = []
        rows_10min = []
        next_day_user: dict = {}

        for user in users:
            user_df = day_df[day_df["hostid"] == user].sort_values("ts")
            traffic = [np.nan] * NUM_MINUTES

            for _, row in user_df.iterrows():
                rounded = _round_to_minute(row["ts"].to_pydatetime())
                slot = rounded.strftime("%Y-%m-%d %H:%M")

                if slot not in slot_to_idx:
                    next_day_user[user] = float(row["traffic"]) * GRANULARITY / BYTE
                    continue

                idx = slot_to_idx[slot]
                if not np.isnan(traffic[idx]):
                    continue  # DST duplicate: keep first occurrence

                traffic[idx] = float(row["traffic"]) * GRANULARITY / BYTE

            if np.isnan(traffic[0]) and user in last_day_user:
                traffic[0] = last_day_user[user]

            for i in range(NUM_MINUTES - 1):
                if np.isnan(traffic[i]):
                    traffic[i + 1] = np.nan

            if np.any(~np.isnan(traffic)):
                record = [user, current_day] + traffic
                rows_all.append(record)
                if _compute_max_consec_nan(pd.Series(traffic)) <= 10:
                    rows_10min.append(record)

        last_day_user = next_day_user

        if rows_all:
            pd.DataFrame(rows_all, columns=_COL_NAMES).to_csv(
                out_daily / f"{_DAILY_PREFIX}{current_day}.csv", index=False
            )
            pd.DataFrame(rows_10min, columns=_COL_NAMES).to_csv(
                out_10min / f"{_10MIN_PREFIX}{current_day}.csv", index=False
            )

        print(f"  {current_day}: {len(rows_all)} users (allSeries), {len(rows_10min)} (10min)")


def align(measure: str):
    raw_dir = RAW_DIR / measure
    raw_files = sorted(raw_dir.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")

    out_daily = DAILY_DIR / measure / "allSeries"
    out_10min = DAILY_DIR / measure / "10min"

    print(f"=== Stage 2: align [{measure}] — {len(raw_files)} file(s) ===")
    for f in raw_files:
        print(f"\n--- {f.name} ---")
        _align_one_file(f, out_daily, out_10min)
    print("Done.\n")


# ---------------------------------------------------------------------------
# Stage 3: stats — port of get_allSeries_stats.py (Python 3 + modern pandas)
# ---------------------------------------------------------------------------

def compute_stats(measure: str) -> pd.DataFrame:
    daily_dir = DAILY_DIR / measure / "allSeries"
    day_files = sorted(daily_dir.glob(f"{_DAILY_PREFIX}*.csv"))
    if not day_files:
        raise FileNotFoundError(f"No daily series found in {daily_dir}")

    combined = pd.concat([pd.read_csv(f) for f in day_files], ignore_index=True)
    traffic = combined.iloc[:, 2:]
    stats_df = pd.DataFrame({
        "day": combined.iloc[:, 1],
        "user": combined.iloc[:, 0],
        "nan_total": traffic.isna().sum(axis=1).astype(int),
        "max_consec": traffic.apply(_compute_max_consec_nan, axis=1),
    })

    out_path = DAILY_DIR / measure / "allSeries_stats.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(stats_df)} rows)")
    return stats_df


# ---------------------------------------------------------------------------
# Stage 4: filter — port of filter_allSeries.py (Python 3 + modern pandas)
# ---------------------------------------------------------------------------

def _passes_filter(row: pd.Series) -> bool:
    series = row.iloc[2:]
    return (
        _compute_max_consec_nan(series) <= MAX_CONSEC_NAN
        and int(series.isna().sum()) < MAX_TOTAL_NAN
    )


def _iso_week_label(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%G-W%V")


def _load_selection(measure: str) -> dict | None:
    """
    Load optional selected_allSeries.csv mapping filenames to (week, flag).
    Accepts both 'filename' and 'day_file' as the key column (legacy support).
    Returns None if the file doesn't exist.
    """
    path = DAILY_DIR / measure / "selected_allSeries.csv"
    if not path.exists():
        return None
    sel = pd.read_csv(path)
    if "day_file" in sel.columns:
        sel = sel.rename(columns={"day_file": "filename"})
    sel["filename"] = sel["filename"].str.strip()
    sel["flag"] = sel.get("flag", pd.Series(1, index=sel.index)).fillna(1).astype(int)
    return dict(zip(sel["filename"], zip(sel["week"].astype(str), sel["flag"])))


def filter_series(measure: str):
    """
    Stage 4: quality-filter daily series and group into weekly CSVs.

    Thresholds (from SUESTE_OLD/filter_allSeries.py):
      max_consec_NaN ≤ 180 min  (≤ 3 consecutive hours missing)
      nan_total      < 720 min  (< 50% of day missing)

    Week assignment: ISO calendar week, or from selected_allSeries.csv if present.
    Days with flag=0 in the selection file are skipped entirely.

    Output: data/input/{measure}/series_giga_filtered_{start}_{end}.csv
    No extra index column — train_model.py reads these directly without skiprows.
    """
    daily_dir = DAILY_DIR / measure / "allSeries"
    out_dir = INPUT_DIR / measure
    out_dir.mkdir(parents=True, exist_ok=True)

    day_files = sorted(daily_dir.glob(f"{_DAILY_PREFIX}*.csv"))
    if not day_files:
        raise FileNotFoundError(f"No daily series found in {daily_dir}")

    selection = _load_selection(measure)

    weeks: dict[str, list[Path]] = {}
    for f in day_files:
        date_str = f.stem.replace(_DAILY_PREFIX, "")
        if selection is not None and f.name not in selection:
            print(f"  Skipping {f.name} (not in selection file)")
            continue
        if selection is not None and selection[f.name][1] != 1:
            print(f"  Skipping {f.name} (flag=0)")
            continue
        week_label = _iso_week_label(date_str) if selection is None else selection[f.name][0]
        weeks.setdefault(week_label, []).append(f)

    print(f"=== Stage 4: filter [{measure}] — {len(weeks)} week(s) ===")
    for week_label, files in sorted(weeks.items()):
        frames = []
        for f in sorted(files):
            df = pd.read_csv(f)
            before = len(df)
            df = df[df.apply(_passes_filter, axis=1)].reset_index(drop=True)
            print(f"  {f.name}: {before} → {len(df)} rows")
            frames.append(df)

        if not frames:
            continue

        week_df = pd.concat(frames, ignore_index=True)
        days = sorted(week_df["day"].unique())
        out_path = out_dir / f"series_giga_filtered_{days[0]}_{days[-1]}.csv"
        week_df.to_csv(out_path, index=False)
        print(f"  => {out_path.name} ({len(week_df)} rows, {len(days)} days)\n")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw traffic data for PARAFAC training."
    )
    parser.add_argument(
        "stage",
        choices=["extract", "align", "stats", "filter", "all"],
    )
    parser.add_argument(
        "--measure", choices=["down", "up", "both"], required=True,
        help="Traffic direction. Use 'both' to process down and up in sequence.",
    )
    args = parser.parse_args()

    measures = ["down", "up"] if args.measure == "both" else [args.measure]

    for measure in measures:
        if args.stage == "extract":
            extract(measure)
        elif args.stage == "align":
            align(measure)
        elif args.stage == "stats":
            print(f"=== Stage 3: stats [{measure}] ===")
            compute_stats(measure)
        elif args.stage == "filter":
            filter_series(measure)
        elif args.stage == "all":
            align(measure)
            filter_series(measure)


if __name__ == "__main__":
    main()

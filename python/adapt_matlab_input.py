"""
Convert MATLAB-format traffic tables to the CSV format expected by train_model.py.

MATLAB source (series_filter_NaNs/):
  idSamples.csv     — N rows: Var1=day, Var2=mac_address
  traffic_down.csv  — N rows × 1440 cols (Var1…Var1440), values in bits/min
  traffic_up.csv    — same for upload

Output (one file per ISO week, per direction):
  <out_dir>/down/series_giga_filtered_<start>_<end>.csv
  <out_dir>/up/series_giga_filtered_<start>_<end>.csv
  Columns: user_id | day | 00:00 | 00:01 | … | 23:59

Usage:
  python adapt_matlab_input.py \\
      --train-dir SUESTE_OLD/st1/matlab/parafac_coronavirus/tables/models/train/2019-08-19_2019-09-22/series_filter_NaNs \\
      --out-dir   python/data/input
"""

import argparse
from pathlib import Path

import pandas as pd

from preprocess import _MINUTE_COLS


def adapt(train_dir: Path, out_dir: Path):
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)

    print("Loading idSamples.csv …")
    ids = pd.read_csv(train_dir / "idSamples.csv")
    ids.columns = ["day", "user_id"]

    print("Loading traffic_down.csv …")
    traffic_down = pd.read_csv(train_dir / "traffic_down.csv")
    traffic_down.columns = _MINUTE_COLS

    print("Loading traffic_up.csv …")
    traffic_up = pd.read_csv(train_dir / "traffic_up.csv")
    traffic_up.columns = _MINUTE_COLS

    ids = ids.reset_index(drop=True)
    down = pd.concat([ids[["user_id", "day"]], traffic_down.reset_index(drop=True)], axis=1)
    up   = pd.concat([ids[["user_id", "day"]], traffic_up.reset_index(drop=True)],   axis=1)

    # Group rows by ISO week and write one CSV per week per direction
    week_label = (
        pd.to_datetime(down["day"]).dt.isocalendar()
        .apply(lambda r: f"{r.year}-W{r.week:02d}", axis=1)
    )

    print(f"\nFound {week_label.nunique()} ISO weeks — splitting …\n")
    for label in sorted(week_label.unique()):
        mask = week_label == label
        week_down = down[mask].reset_index(drop=True)
        week_up   = up[mask].reset_index(drop=True)

        days = sorted(week_down["day"].unique())
        start, end = days[0], days[-1]
        filename = f"series_giga_filtered_{start}_{end}.csv"

        for direction, df in [("down", week_down), ("up", week_up)]:
            path = out_dir / direction / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)

        print(f"  {label}  {start} → {end}  ({len(days)} days, {len(week_down)} UD pairs)")
        print(f"    → {out_dir}/down/{filename}")
        print(f"    → {out_dir}/up/{filename}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MATLAB traffic tables to train_model.py CSV format."
    )
    parser.add_argument(
        "--train-dir", required=True,
        help="Path to the series_filter_NaNs/ directory",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Base output directory — down/ and up/ are created inside it",
    )
    args = parser.parse_args()
    adapt(args.train_dir, args.out_dir)


if __name__ == "__main__":
    main()

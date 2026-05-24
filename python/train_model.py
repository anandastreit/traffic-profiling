"""
PARAFAC traffic model — Python port of the MATLAB training scripts.

Replaces:
  models_allSeries_filtered_weekDays.m  -> mode "per_week"
  models_allSeries_filtered_allDays.m   -> mode "all_days"

Input CSVs (place in data/input/down/ and data/input/up/):
  One CSV per week, sorted alphabetically = week order.
  Format: user_id | day | minute_0 | ... | minute_1439
  (MATLAB readtable drops an extra header row, so Python skips row index 1.)

Output CSVs (written to data/output/<mode>/):
  modeA_<label>.csv  - user loadings     (n_users x n_components)
  modeB_<label>.csv  - time loadings     (1440 x n_components)
  modeC_<label>.csv  - day loadings      (n_days x n_components)
  modeD_<label>.csv  - direction loading (2 x n_components)
  ids_<label>.csv    - user_id and day labels matching modeA rows

Usage:
  python train_model.py per_week
  python train_model.py all_days
  python train_model.py both --components 3 --tol 1e-10
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import non_negative_parafac_hals

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_COMP = 3
TOLERANCE = 1e-10
SEED = 123
NUM_MINUTES = 1440  # minutes per day

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
DOWN_DIR = os.path.join(DATA_DIR, "input", "down")
UP_DIR = os.path.join(DATA_DIR, "input", "up")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


# ---------------------------------------------------------------------------
# Normalization — mirrors MATLAB nlog10.m
# ---------------------------------------------------------------------------

def nlog10(array: np.ndarray) -> np.ndarray:
    """log10(x + 1) — equivalent to MATLAB nlog10(array, 1)."""
    return np.log10(array + 1)


def nlog10_inverse(array: np.ndarray) -> np.ndarray:
    """Inverse: 10^x - 1 — equivalent to MATLAB nlog10(array, -1)."""
    return (10.0 ** array) - 1.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_week_csv(filepath: str):
    """
    Load one weekly traffic CSV.

    MATLAB does readtable() then drops row 1 (tbl(1,:)=[]), meaning the file
    has an extra row after the header that must be skipped.

    Returns:
        ids  : DataFrame with columns ['user_id', 'day']
        data : ndarray of shape (n_samples, NUM_MINUTES)
    """
    df = pd.read_csv(filepath, skiprows=[1])  # skip the extra row MATLAB drops
    ids = df.iloc[:, :2].copy()
    ids.columns = ["user_id", "day"]
    data = df.iloc[:, 2:].values.astype(float)
    return ids, data


# ---------------------------------------------------------------------------
# Tensor construction — mirrors MATLAB build loop
# ---------------------------------------------------------------------------

def build_week_tensor(ids_down: pd.DataFrame, data_down: np.ndarray,
                      ids_up: pd.DataFrame, data_up: np.ndarray):
    """
    Build a 4D tensor for one week: (users, minutes, days, direction).
    direction 0 = download, direction 1 = upload.
    Missing user/day combinations are filled with NaN.

    Mirrors the per-week loop in models_allSeries_filtered_weekDays.m.
    """
    # Intersect users present in both down and up (same as MATLAB intersect)
    key_down = ids_down.set_index(["user_id", "day"]).index
    key_up = ids_up.set_index(["user_id", "day"]).index
    common_keys = key_down.intersection(key_up)

    mask_down = key_down.isin(common_keys)
    mask_up = key_up.isin(common_keys)

    ids_down = ids_down[mask_down].reset_index(drop=True)
    ids_up = ids_up[mask_up].reset_index(drop=True)
    data_down = data_down[mask_down]
    data_up = data_up[mask_up]

    # Normalize
    norm_down = nlog10(data_down)
    norm_up = nlog10(data_up)

    users = ids_down["user_id"].unique()
    days = ids_down["day"].unique()

    user_idx = {u: i for i, u in enumerate(users)}
    day_idx = {d: i for i, d in enumerate(days)}

    X = np.full((len(users), NUM_MINUTES, len(days), 2), np.nan)

    for row_i in range(len(ids_down)):
        ui = user_idx[ids_down.loc[row_i, "user_id"]]
        di = day_idx[ids_down.loc[row_i, "day"]]
        X[ui, :, di, 0] = norm_down[row_i]
        X[ui, :, di, 1] = norm_up[row_i]

    print(f"  Intersection: {len(users)} users, {len(days)} days")
    print(f"  Tensor shape: {X.shape}")
    return X, users, days


# ---------------------------------------------------------------------------
# PARAFAC — mirrors MATLAB parafac() call with const=[2,2,2,2]
# ---------------------------------------------------------------------------

def run_parafac(X: np.ndarray, n_components: int = NUM_COMP,
                tol: float = TOLERANCE, seed: int = SEED):
    """
    Non-negative PARAFAC decomposition.

    Equivalent to MATLAB:
        rng(123,'twister');
        parafac(X, NUM_COMP, Options, const)
    with Options(1)=tol, Options(3)=0 (no plots), Options(4)=1 (default scaling)
    and const=[2 2 2 2] (non-negativity on all modes).

    NOTE on NaN handling: MATLAB's N-way toolbox skips NaN entries during ALS
    updates. Here we replace NaN with 0 before decomposition, which is a
    simplification. Validate against MATLAB outputs to assess impact.
    """
    tensor = tl.tensor(np.nan_to_num(X, nan=0.0))
    cp = non_negative_parafac_hals(
        tensor,
        rank=n_components,
        n_iter_max=10000,
        tol=tol,
        random_state=seed,
        verbose=False,
    )
    return cp


# ---------------------------------------------------------------------------
# Output saving — matches MATLAB CSV format (final_model1, final_model2, ...)
# ---------------------------------------------------------------------------

def save_model(cp, label: str, users: np.ndarray, days: np.ndarray,
               output_dir: str):
    """
    Save all factor matrices and ID metadata as CSVs.
    Column names match MATLAB output: final_model1, final_model2, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    n_comp = cp.factors[0].shape[1]
    col_names = [f"final_model{i+1}" for i in range(n_comp)]

    mode_names = ["modeA", "modeB", "modeC", "modeD"]
    for factor, mode in zip(cp.factors, mode_names):
        df = pd.DataFrame(factor, columns=col_names)
        path = os.path.join(output_dir, f"{mode}_{label}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")

    # Save ID metadata (needed to match modeA rows to users/days)
    ids_df = pd.DataFrame({"user_id": users})
    ids_path = os.path.join(output_dir, f"ids_users_{label}.csv")
    ids_df.to_csv(ids_path, index=False)

    days_df = pd.DataFrame({"day": days})
    days_path = os.path.join(output_dir, f"ids_days_{label}.csv")
    days_df.to_csv(days_path, index=False)

    print(f"  Saved: {ids_path}")
    print(f"  Saved: {days_path}")


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def train_per_week(output_base: str, n_components: int, tol: float):
    """
    One independent PARAFAC model per week.
    Mirrors models_allSeries_filtered_weekDays.m
    """
    down_files = sorted(glob.glob(os.path.join(DOWN_DIR, "*.csv")))
    up_files = sorted(glob.glob(os.path.join(UP_DIR, "*.csv")))

    if not down_files:
        raise FileNotFoundError(f"No CSV files found in {DOWN_DIR}")
    if len(down_files) != len(up_files):
        raise ValueError(
            f"Mismatch: {len(down_files)} down files vs {len(up_files)} up files"
        )

    for week_num, (down_path, up_path) in enumerate(zip(down_files, up_files), start=1):
        label = f"week{week_num:02d}"
        print(f"\n=== Week {week_num}: {os.path.basename(down_path)} ===")

        ids_down, data_down = load_week_csv(down_path)
        ids_up, data_up = load_week_csv(up_path)
        print(f"  Down: {len(ids_down)} rows | Up: {len(ids_up)} rows")

        X, users, days = build_week_tensor(ids_down, data_down, ids_up, data_up)

        print(f"  Fitting PARAFAC (rank={n_components}, tol={tol})...")
        cp = run_parafac(X, n_components=n_components, tol=tol)

        out_dir = os.path.join(output_base, "per_week", label)
        save_model(cp, label, users, days, out_dir)


def train_all_days(output_base: str, n_components: int, tol: float):
    """
    One PARAFAC model on the full cumulative tensor (all weeks combined).
    Mirrors models_allSeries_filtered_allDays.m
    """
    down_files = sorted(glob.glob(os.path.join(DOWN_DIR, "*.csv")))
    up_files = sorted(glob.glob(os.path.join(UP_DIR, "*.csv")))

    if not down_files:
        raise FileNotFoundError(f"No CSV files found in {DOWN_DIR}")
    if len(down_files) != len(up_files):
        raise ValueError(
            f"Mismatch: {len(down_files)} down files vs {len(up_files)} up files"
        )

    X_all = None
    all_users = []
    all_days = []

    for week_num, (down_path, up_path) in enumerate(zip(down_files, up_files), start=1):
        print(f"\n--- Week {week_num}: {os.path.basename(down_path)} ---")

        ids_down, data_down = load_week_csv(down_path)
        ids_up, data_up = load_week_csv(up_path)
        print(f"  Down: {len(ids_down)} rows | Up: {len(ids_up)} rows")

        X_week, users_week, days_week = build_week_tensor(
            ids_down, data_down, ids_up, data_up
        )

        new_users = [u for u in users_week if u not in all_users]
        print(f"  New users this week: {len(new_users)}")

        if X_all is None:
            X_all = X_week.copy()
            all_users = list(users_week)
            all_days = list(days_week)
        else:
            n_existing_users = len(all_users)
            n_existing_days = len(all_days)
            n_new_days = len(days_week)
            n_new_users = len(new_users)

            # Extend day axis for existing users
            X_all = np.concatenate(
                [X_all, np.full((n_existing_users, NUM_MINUTES, n_new_days, 2), np.nan)],
                axis=2,
            )

            # Add rows for brand-new users (NaN for all days so far)
            if n_new_users > 0:
                X_all = np.concatenate(
                    [X_all, np.full((n_new_users, NUM_MINUTES,
                                     n_existing_days + n_new_days, 2), np.nan)],
                    axis=0,
                )
                all_users.extend(new_users)

            # Fill this week's data into the global tensor
            user_idx_global = {u: i for i, u in enumerate(all_users)}
            day_start = n_existing_days
            for di, day in enumerate(days_week):
                for ui, user in enumerate(users_week):
                    gi = user_idx_global[user]
                    X_all[gi, :, day_start + di, :] = X_week[ui, :, di, :]

            all_days.extend(list(days_week))

        print(f"  Global tensor shape so far: {X_all.shape}")

    print(f"\n=== Fitting PARAFAC on full tensor {X_all.shape} "
          f"(rank={n_components}, tol={tol}) ===")
    cp = run_parafac(X_all, n_components=n_components, tol=tol)

    out_dir = os.path.join(output_base, "all_days")
    save_model(cp, "all_days", np.array(all_users), np.array(all_days), out_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PARAFAC on network traffic tensor data."
    )
    parser.add_argument(
        "mode",
        choices=["per_week", "all_days", "both"],
        help=(
            "per_week: one model per week (mirrors weekDays.m); "
            "all_days: one model on all weeks (mirrors allDays.m); "
            "both: run both."
        ),
    )
    parser.add_argument(
        "--components", type=int, default=NUM_COMP,
        help=f"Number of PARAFAC components (default: {NUM_COMP})",
    )
    parser.add_argument(
        "--tol", type=float, default=TOLERANCE,
        help=f"Convergence tolerance (default: {TOLERANCE})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Base output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if args.mode in ("per_week", "both"):
        print("\n====== PER-WEEK MODELS ======")
        train_per_week(args.output, args.components, args.tol)

    if args.mode in ("all_days", "both"):
        print("\n====== ALL-DAYS MODEL ======")
        train_all_days(args.output, args.components, args.tol)


if __name__ == "__main__":
    main()

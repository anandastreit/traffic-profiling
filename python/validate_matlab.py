"""
Validate Python PARAFAC outputs against MATLAB reference.

== What is being compared ==

MATLAB side (matlab/data/loadings_all.csv):
  Fatores extraídos de arquivos .mat gerados no MATLAB (Windows).
  Dois tipos para o período de treinamento 2019-08-19_2019-09-22:

  type=weekly  (weeks 1-5): 5 modelos semanais independentes
    Origem: weekModel5_..._seed456_v2.mat  (seed=456, nComp=5)
    Comparado com: python/data/output/per_week/week01..05/

  type=reference: 1 modelo treinado em TODAS as 5 semanas juntas
    Origem: models_1-08_..._noWeights.mat -> Model5  (58048 usuarios)
    Comparado com: python/data/output/all_days/

Python side:
  per_week/  — train_model.py per_week  (seed=123, nComp=5)
  all_days/  — train_model.py all_days  (seed=123, nComp=5)

== Metrica ==

Tucker Congruence Coefficient (TCC) = cosine similarity entre colunas
normalizadas. Robusto a permutacao de colunas e redistribuicao de escala
entre modos (ambiguidades normais do PARAFAC).

  TCC >= 0.95 = excelente (modelos equivalentes)
  TCC  0.85-0.95 = bom
  TCC < 0.85 = falhou

Usage:
  python validate_matlab.py
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
LOADINGS_CSV = os.path.join(_HERE, "..", "matlab", "data", "loadings_all.csv")
PER_WEEK_DIR = os.path.join(_HERE, "data", "output", "per_week")
ALL_DAYS_DIR = os.path.join(_HERE, "data", "output_tol1e6", "all_days")

TRAINING_SOURCE = "2019-08-19_2019-09-22"
N_COMP = 5
COMP_COLS = [f"comp_{i+1}" for i in range(N_COMP)]

WEEK_MAP = {
    "1": "week01",
    "2": "week02",
    "3": "week03",
    "4": "week04",
    "5": "week05",
}


def tcc_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each pair of columns of a and b (shape m×k)."""
    a_n = a / np.linalg.norm(a, axis=0, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=0, keepdims=True)
    return a_n.T @ b_n  # (k×k)


def best_alignment(tcc_mat: np.ndarray):
    """Hungarian algorithm: find column permutation that maximises sum of TCC."""
    row_ind, col_ind = linear_sum_assignment(-tcc_mat)
    return row_ind, col_ind, tcc_mat[row_ind, col_ind]


def _report(label: str, mat: np.ndarray, py: np.ndarray):
    if mat.shape != py.shape:
        print(
            f"  {label}: shape mismatch  matlab={mat.shape}  python={py.shape}"
        )
        return
    tcc = tcc_matrix(mat, py)
    row_ind, col_ind, tcc_vals = best_alignment(tcc)
    alignment = [f"{r}→{c}" for r, c in zip(row_ind, col_ind)]
    tcc_str = "  ".join(f"{v:.3f}" for v in tcc_vals)
    mean_tcc = tcc_vals.mean()
    flag = (
        ""
        if mean_tcc >= 0.95
        else "  ⚠ <0.95"
        if mean_tcc >= 0.85
        else "  ✗ <0.85"
    )
    print(
        f"  {label}  align=[{', '.join(alignment)}]"
        f"  TCC=[{tcc_str}]  mean={mean_tcc:.4f}{flag}"
    )


def _load_matlab(df: pd.DataFrame, type_: str, week: str, mode: str) -> np.ndarray:
    subset = df[
        (df["source"] == TRAINING_SOURCE)
        & (df["type"] == type_)
        & (df["week"] == week)
        & (df["mode"] == mode)
    ].sort_values("row")
    return subset[COMP_COLS].values.astype(float)


def _load_python(folder: str, filename: str) -> np.ndarray:
    return pd.read_csv(os.path.join(folder, filename)).values.astype(float)


# ---------------------------------------------------------------------------
# Per-week validation: Python per_week vs MATLAB type=weekly
# ---------------------------------------------------------------------------

def validate_per_week(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("PER-WEEK: Python per_week vs MATLAB type=weekly")
    print("  (5 modelos independentes, um por semana)")
    print("=" * 60)

    for week_num, py_label in WEEK_MAP.items():
        week_date = df[
            (df["source"] == TRAINING_SOURCE)
            & (df["type"] == "weekly")
            & (df["week"] == week_num)
        ]["week_date"].iloc[0]
        print(f"\n--- Week {week_num} ({week_date}) ---")

        for mode_letter, mode_prefix in [("B", "modeB"), ("C", "modeC")]:
            mat = _load_matlab(df, "weekly", week_num, mode_letter)
            py = _load_python(
                os.path.join(PER_WEEK_DIR, py_label),
                f"{mode_prefix}_{py_label}.csv",
            )
            _report(mode_prefix, mat, py)


# ---------------------------------------------------------------------------
# Reference model validation: Python all_days vs MATLAB type=reference
# ---------------------------------------------------------------------------

def validate_reference(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("REFERENCE: Python all_days vs MATLAB type=reference")
    print("  (1 modelo treinado em todas as 5 semanas juntas)")
    print("=" * 60)

    all_days_modeB = os.path.join(ALL_DAYS_DIR, "modeB_all_days.csv")
    all_days_modeC = os.path.join(ALL_DAYS_DIR, "modeC_all_days.csv")

    if not os.path.exists(all_days_modeB):
        print(
            "\n  [SKIP] all_days output not found."
            " Run: python train_model.py all_days"
        )
        return

    print()
    for mode_letter, py_path in [("B", all_days_modeB), ("C", all_days_modeC)]:
        mat = _load_matlab(df, "reference", "reference", mode_letter)
        py = pd.read_csv(py_path).values.astype(float)
        _report(f"mode{mode_letter}", mat, py)


# ---------------------------------------------------------------------------

def main():
    print(f"Loading {LOADINGS_CSV} …")
    df = pd.read_csv(LOADINGS_CSV, low_memory=False)
    df["week"] = df["week"].astype(str)

    validate_per_week(df)
    validate_reference(df)

    print("\n\nTCC >= 0.95 = Python and MATLAB models are equivalent.")


if __name__ == "__main__":
    main()

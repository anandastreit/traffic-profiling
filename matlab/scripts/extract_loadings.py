"""
Extrai os loadings PARAFAC (A, B, C) e salva em CSV unico.

Fontes:
  1. Modelo de referencia (type=reference): lido dos CSVs canonicos em
       tables/models/train/2019-08-19_2019-09-22/model5_mode{1,2,3}_trafficDownUp_train_noWeights*.csv
     Origem: models_1-08_..._noWeights.mat -> Model5 via fac2let()
  2. Modelos semanais (type=weekly): lidos dos
       weekModel5_trafficDownUp_*_seed456_v2.mat  (NCOMP=5, seed=456)

Colunas do CSV de saida:
  source     : periodo de datas (ex: "2019-08-19_2019-09-22")
  type       : "reference" ou "weekly"
  week       : "reference" ou numero da semana (1, 2, 3...)
  week_date  : "reference" ou datas reais da semana (ex: "2019-08-19_2019-08-25")
  mode       : "A" (usuarios), "B" (tempo), "C" (metrica: down/up)
  row        : indice da linha no loading (0-based)
                - Modo A: indice do usuario
                - Modo B: slot de tempo (minuto do dia, 0-1439)
                - Modo C: 0 = traffic_down, 1 = traffic_up
  comp_1..comp_5 : valores dos 5 componentes PARAFAC
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import scipy.io
import h5py

FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(FOLDER, "loadings_all.csv")
NCOMP_WEEKLY = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loading_to_rows(matrix, source, type_label, week, week_date, mode, ncomp):
    """Converte uma matriz de loadings (n x ncomp) em lista de dicts."""
    rows = []
    comp_cols = [f"comp_{c+1}" for c in range(ncomp)]
    for i, row_vals in enumerate(matrix):
        d = {
            "source": source,
            "type": type_label,
            "week": week,
            "week_date": week_date,
            "mode": mode,
            "row": i,
        }
        for c in range(ncomp):
            d[f"comp_{c+1}"] = float(row_vals[c])
        rows.append(d)
    return rows


# ---------------------------------------------------------------------------
# Leitura dos weekModel5 (formato v7, scipy)
# ---------------------------------------------------------------------------

def read_weekmodel(filepath):
    """Le um weekModel5_*_seed456_v2.mat e retorna lista de dicts."""
    filename = os.path.basename(filepath)
    m = re.search(r'weekModel5_trafficDownUp_(.+?)_seed456_v2\.mat', filename)
    if not m:
        print(f"  [AVISO] Nome inesperado: {filename}, pulando.")
        return []
    source = m.group(1)

    print(f"  Lendo {filename}...")
    try:
        mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    except Exception as e:
        print(f"  [ERRO scipy] {e}")
        return []

    model_struct = mat.get("Model")
    if model_struct is None:
        print(f"  [AVISO] Variavel 'Model' nao encontrada.")
        return []

    rows = []
    week_num = 1
    while hasattr(model_struct, f"week{week_num}"):
        week_obj = getattr(model_struct, f"week{week_num}")
        cell = week_obj.model  # numpy array (3,) com A, B, C

        A = np.atleast_2d(np.array(cell.flat[0], dtype=float))
        B = np.atleast_2d(np.array(cell.flat[1], dtype=float))
        C = np.atleast_2d(np.array(cell.flat[2], dtype=float))

        # data real da semana (ex: "2019-08-19_2019-08-25")
        week_date = str(week_obj.date).strip() if hasattr(week_obj, "date") else "?"

        for arr, mode_label in [(A, "A"), (B, "B"), (C, "C")]:
            if arr.shape[1] != NCOMP_WEEKLY:
                arr = arr.T
            rows.extend(loading_to_rows(arr, source, "weekly", week_num, week_date, mode_label, NCOMP_WEEKLY))

        week_num += 1

    print(f"    {week_num-1} semanas, {len(rows)} linhas")
    return rows


# ---------------------------------------------------------------------------
# Leitura dos modelos de referencia (formato v7.3 HDF5, h5py)
# ---------------------------------------------------------------------------

def read_reference_model(filepath):
    """Le models_1-08_*_v2.mat (v7.3) e extrai todos os nCompN disponiveis."""
    filename = os.path.basename(filepath)
    m = re.search(r'models_\d+-\d+_trafficDownUp_(.+?)_noWeights', filename)
    source = m.group(1) if m else "reference"
    print(f"  Lendo modelo de referencia: {filename}...")

    rows = []
    try:
        with h5py.File(filepath, "r") as f:
            dataset = f["Dataset"]
            ncomp_keys = sorted([k for k in dataset.keys() if k.startswith("nComp")])
            print(f"    Componentes disponiveis: {ncomp_keys}")

            for ncomp_key in ncomp_keys:
                ncomp = int(ncomp_key.replace("nComp", ""))
                fit = float(dataset[ncomp_key]["fit"][0, 0])
                model_refs = dataset[ncomp_key]["model"]  # (3,1) de refs HDF5

                # HDF5/MATLAB v7.3 transposiciona matrizes
                A = np.array(f[model_refs[0, 0]], dtype=float).T  # (n_users x ncomp)
                B = np.array(f[model_refs[1, 0]], dtype=float).T  # (n_time  x ncomp)
                C = np.array(f[model_refs[2, 0]], dtype=float).T  # (2       x ncomp)

                type_label = f"reference_ncomp{ncomp}"
                print(f"    nComp{ncomp}: fit={fit:.2f}% | A={A.shape} B={B.shape} C={C.shape}")

                for arr, mode_label in [(A, "A"), (B, "B"), (C, "C")]:
                    if arr.shape[1] != ncomp:
                        arr = arr.T
                    rows.extend(loading_to_rows(arr, source, type_label, "reference", "reference", mode_label, ncomp))

    except Exception as e:
        print(f"  [ERRO h5py] {e}")
        return []

    print(f"    Total: {len(rows)} linhas")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_rows = []

    # 1. Modelo de referencia — lido dos CSVs canonicos
    REF_SOURCE = "2019-08-19_2019-09-22"
    REF_DIR = os.path.join(FOLDER, "tables", "models", "train", REF_SOURCE)
    ref_csvs = {
        "A": os.path.join(REF_DIR, "model5_mode1_trafficDownUp_train_noWeights_filter_NaNs.csv"),
        "B": os.path.join(REF_DIR, "model5_mode2_trafficDownUp_train_noWeights.csv"),
        "C": os.path.join(REF_DIR, "model5_mode3_trafficDownUp_train_noWeights.csv"),
    }
    print("Lendo modelo de referencia (CSVs canonicos)...")
    for mode_label, csv_path in ref_csvs.items():
        arr = pd.read_csv(csv_path, header=None).values
        print(f"  {mode_label}: {arr.shape}")
        for i, row_vals in enumerate(arr):
            d = {"source": REF_SOURCE, "type": "reference", "week": "reference",
                 "week_date": "reference", "mode": mode_label, "row": i}
            for c in range(NCOMP_WEEKLY):
                d[f"comp_{c+1}"] = float(row_vals[c])
            all_rows.append(d)

    # 2. Modelos semanais (weekModel5 seed456_v2, nComp=5)
    weekly_pattern = os.path.join(FOLDER, "weekModel5_trafficDownUp_*_seed456_v2.mat")
    weekly_files = sorted(glob.glob(weekly_pattern))
    print(f"\nEncontrados {len(weekly_files)} arquivos semanais seed456_v2.")
    for fp in weekly_files:
        all_rows.extend(read_weekmodel(fp))

    if not all_rows:
        print("Nenhum dado extraido. Verifique os arquivos e dependencias.")
        return

    df = pd.DataFrame(all_rows)

    # Colunas de componentes dinamicas (max ncomp encontrado)
    max_comp = max(int(c.replace("comp_", "")) for c in df.columns if c.startswith("comp_"))
    comp_cols = [f"comp_{i}" for i in range(1, max_comp + 1)]
    col_order = ["source", "type", "week", "week_date", "mode", "row"] + comp_cols
    df = df.reindex(columns=col_order)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV salvo em: {OUTPUT_CSV}")
    print(f"Total de linhas: {len(df):,}")
    print("\nResumo por tipo:")
    print(df.groupby(["type", "mode"])["row"].count().to_string())


if __name__ == "__main__":
    main()

"""Proper rank sweep: real algorithm (masked non_negative_parafac), run to
genuine convergence (tol=1e-6, generous n_iter_max), one 2019 week + two WAN
weeks, ranks 3-7.

Durable + resumable by design (see project memory
feedback_longrun_durable_storage.md, written after a reboot wiped an earlier
/tmp-based run of this same sweep):
  - All output lives under this same directory, inside the real project tree
    (gitignored via python/data/exploratory/**/*), never /tmp or a scratchpad.
  - Each (dataset, rank) unit is skipped if its info_*.json already exists,
    so re-running this script after any interruption (crash, reboot, kill)
    picks up exactly where it left off.
  - Each unit's fit is wrapped in try/except: a failure on one unit is
    logged to failed_<name>_rank<r>.json and the queue continues, rather
    than the whole sweep dying.

Run (survives terminal close): nohup + background, or resume by just
re-running the same command — already-done units are skipped instantly.
"""
import sys, os, time, json, traceback
sys.path.insert(0, "/home/localuser/Documents/projects/parafac_traffic/python")
os.chdir("/home/localuser/Documents/projects/parafac_traffic/python")
import numpy as np
from train_model import load_week_csv, build_week_tensor, run_parafac

OUT = os.path.dirname(os.path.abspath(__file__))  # this directory itself

DATASETS = {
    "2019wk1": ("data/input/down/series_giga_filtered_2019-08-19_2019-08-25.csv",
               "data/input/up/series_giga_filtered_2019-08-19_2019-08-25.csv"),
    "wan10": ("data/input_wan/down/series_wan_filtered_2026-06-01_2026-06-07.csv",
             "data/input_wan/up/series_wan_filtered_2026-06-01_2026-06-07.csv"),
    "wan21": ("data/input_wan/down/series_wan_filtered_2026-08-17_2026-08-23.csv",
             "data/input_wan/up/series_wan_filtered_2026-08-17_2026-08-23.csv"),
}
RANKS = [3, 4, 5, 6, 7]

queue = [(name, r) for name in DATASETS for r in RANKS]
print(f"{len(queue)} fits in queue", flush=True)

tensors = {}  # lazy-load per dataset, only if at least one unit for it is pending


def get_tensor(name):
    if name not in tensors:
        dp, up = DATASETS[name]
        ids_d, data_d = load_week_csv(dp)
        ids_u, data_u = load_week_csv(up)
        X, ids_ud = build_week_tensor(ids_d, data_d, ids_u, data_u)
        tensors[name] = (X, ids_ud)
        print(f"loaded {name}: tensor shape {X.shape}", flush=True)
        ids_path = f"{OUT}/ids_ud_{name}.csv"
        if not os.path.exists(ids_path):
            ids_ud.to_csv(ids_path, index=False)
    return tensors[name]


for name, rank in queue:
    out_a = f"{OUT}/modeA_{name}_rank{rank}.npy"
    out_b = f"{OUT}/modeB_{name}_rank{rank}.npy"
    out_c = f"{OUT}/modeC_{name}_rank{rank}.npy"
    out_json = f"{OUT}/info_{name}_rank{rank}.json"
    fail_json = f"{OUT}/failed_{name}_rank{rank}.json"
    if os.path.exists(out_json):
        print(f"SKIP {name} rank{rank} (already done)", flush=True)
        continue
    print(f"=== {name} rank{rank} ===", flush=True)
    try:
        X, ids_ud = get_tensor(name)
        t0 = time.time()
        cp, rec_errors, converged = run_parafac(
            X, n_components=rank, tol=1e-6, seed=123)
        dt = time.time() - t0
        A, B, C = [np.asarray(f) for f in cp.factors]  # save ALL three modes
        np.save(out_a, A)
        np.save(out_b, B)
        np.save(out_c, C)
        info = {
            "name": name, "rank": rank, "n_iter": len(rec_errors),
            "converged": converged, "final_error": float(rec_errors[-1]),
            "wall_seconds": dt, "n_ud": int(A.shape[0]),
        }
        with open(out_json, "w") as f:
            json.dump(info, f)
        if os.path.exists(fail_json):
            os.remove(fail_json)  # clear a stale failure record on success
        print(f"  -> {len(rec_errors)} iters, {dt:.0f}s, converged={converged}, "
             f"final_err={rec_errors[-1]:.5f}", flush=True)
    except Exception as e:
        with open(fail_json, "w") as f:
            json.dump({"name": name, "rank": rank, "error": str(e),
                      "traceback": traceback.format_exc()}, f)
        print(f"  !! FAILED {name} rank{rank}: {e}", flush=True)
        continue  # move on to the next unit rather than dying

print("\nALL DONE (or all remaining units attempted)", flush=True)

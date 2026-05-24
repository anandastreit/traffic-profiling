import numpy as np
import pandas as pd
import pytest
from tensorly.cp_tensor import CPTensor

from preprocess import _MINUTE_COLS
from train_model import (
    NUM_MINUTES,
    build_week_tensor,
    load_week_csv,
    nlog10,
    nlog10_inverse,
    run_parafac,
    save_model,
    train_per_week,
)


def _make_week_csv(path, uds, seed=0):
    """Write a synthetic weekly CSV. uds = list of (user_id, day) tuples."""
    rng = np.random.default_rng(seed)
    traffic = rng.integers(0, 1000, size=(len(uds), NUM_MINUTES)).astype(float)
    df = pd.DataFrame(traffic, columns=_MINUTE_COLS)
    df.insert(0, "day", [d for _, d in uds])
    df.insert(0, "user_id", [u for u, _ in uds])
    df.to_csv(path, index=False)


def _make_ids_data(uds, seed=0):
    rng = np.random.default_rng(seed)
    ids = pd.DataFrame(uds, columns=["user_id", "day"])
    data = rng.integers(0, 1000, size=(len(uds), NUM_MINUTES)).astype(float)
    return ids, data


def _make_cp(n_ud=5, n_min=10, rank=2):
    rng = np.random.default_rng(0)
    return CPTensor((np.ones(rank), [
        rng.random((n_ud, rank)),
        rng.random((n_min, rank)),
        rng.random((2, rank)),
    ]))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def test_nlog10_round_trip():
    x = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])
    np.testing.assert_allclose(nlog10_inverse(nlog10(x)), x, rtol=1e-10)


def test_nlog10_zero():
    assert nlog10(np.array([0.0]))[0] == 0.0


def test_nlog10_inverse_zero():
    assert nlog10_inverse(np.array([0.0]))[0] == 0.0


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def test_load_week_csv(tmp_path):
    uds = [("u1", "2018-10-22"), ("u2", "2018-10-22"), ("u3", "2018-10-23")]
    _make_week_csv(tmp_path / "week01.csv", uds)
    ids, data = load_week_csv(str(tmp_path / "week01.csv"))
    assert list(ids.columns) == ["user_id", "day"]
    assert len(ids) == 3
    assert data.shape == (3, NUM_MINUTES)


def test_load_week_csv_values(tmp_path):
    df = pd.DataFrame([[42.0] * NUM_MINUTES], columns=_MINUTE_COLS)
    df.insert(0, "day", ["2018-10-22"])
    df.insert(0, "user_id", ["userA"])
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    ids, data = load_week_csv(str(path))
    assert ids.loc[0, "user_id"] == "userA"
    assert data[0, 0] == 42.0
    assert data[0, -1] == 42.0


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

def test_build_week_tensor_shape():
    uds = [("u1", "d1"), ("u1", "d2"), ("u2", "d1"), ("u2", "d2")]
    ids_down, data_down = _make_ids_data(uds, seed=0)
    ids_up, data_up = _make_ids_data(uds, seed=1)
    X, ids_ud = build_week_tensor(ids_down, data_down, ids_up, data_up)
    assert X.shape == (4, NUM_MINUTES, 2)
    assert len(ids_ud) == 4


def test_build_week_tensor_intersection():
    uds_down = [("u1", "d1"), ("u2", "d1"), ("u3", "d1")]
    uds_up = [("u1", "d1"), ("u2", "d1"), ("u4", "d1")]  # u3 missing, u4 extra
    ids_down, data_down = _make_ids_data(uds_down)
    ids_up, data_up = _make_ids_data(uds_up)
    X, ids_ud = build_week_tensor(ids_down, data_down, ids_up, data_up)
    assert X.shape == (2, NUM_MINUTES, 2)
    assert set(zip(ids_ud["user_id"], ids_ud["day"])) == {("u1", "d1"), ("u2", "d1")}


def test_build_week_tensor_sorted_output():
    uds_down = [("u3", "d1"), ("u1", "d1"), ("u2", "d1")]
    uds_up = [("u2", "d1"), ("u3", "d1"), ("u1", "d1")]
    ids_down, data_down = _make_ids_data(uds_down)
    ids_up, data_up = _make_ids_data(uds_up)
    _, ids_ud = build_week_tensor(ids_down, data_down, ids_up, data_up)
    assert list(ids_ud["user_id"]) == ["u1", "u2", "u3"]


def test_build_week_tensor_no_common_raises():
    ids_down, data_down = _make_ids_data([("u1", "d1")])
    ids_up, data_up = _make_ids_data([("u2", "d1")])
    with pytest.raises(ValueError, match="No common UD pairs"):
        build_week_tensor(ids_down, data_down, ids_up, data_up)


def test_build_week_tensor_normalized():
    uds = [("u1", "d1")]
    data = np.array([[100.0] * NUM_MINUTES])
    ids = pd.DataFrame(uds, columns=["user_id", "day"])
    X, _ = build_week_tensor(ids, data, ids, data.copy())
    expected = np.log10(101.0)
    np.testing.assert_allclose(X[0, :, 0], expected)
    np.testing.assert_allclose(X[0, :, 1], expected)


def test_build_week_tensor_direction_alignment():
    uds = [("u1", "d1")]
    data_down = np.array([[10.0] * NUM_MINUTES])
    data_up = np.array([[20.0] * NUM_MINUTES])
    ids = pd.DataFrame(uds, columns=["user_id", "day"])
    X, _ = build_week_tensor(ids, data_down, ids, data_up)
    np.testing.assert_allclose(X[0, :, 0], np.log10(11.0))  # down
    np.testing.assert_allclose(X[0, :, 1], np.log10(21.0))  # up


# ---------------------------------------------------------------------------
# PARAFAC
# ---------------------------------------------------------------------------

def test_run_parafac_output_shape():
    X = np.random.default_rng(0).random((6, 10, 2))
    cp = run_parafac(X, n_components=2, tol=1e-3, seed=42)
    assert len(cp.factors) == 3
    assert cp.factors[0].shape == (6, 2)
    assert cp.factors[1].shape == (10, 2)
    assert cp.factors[2].shape == (2, 2)


def test_run_parafac_non_negative():
    X = np.random.default_rng(1).random((6, 10, 2))
    cp = run_parafac(X, n_components=2, tol=1e-3, seed=42)
    for factor in cp.factors:
        assert (factor >= 0).all()


def test_run_parafac_with_nan():
    X = np.random.default_rng(2).random((6, 10, 2))
    X[0, 3:6, :] = np.nan
    cp = run_parafac(X, n_components=2, tol=1e-3, seed=42)
    assert len(cp.factors) == 3


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def test_save_model_files(tmp_path):
    cp = _make_cp(n_ud=5, n_min=10, rank=2)
    ids_ud = pd.DataFrame({"user_id": [f"u{i}" for i in range(5)],
                           "day": ["2018-10-22"] * 5})
    save_model(cp, "week01", ids_ud, str(tmp_path))
    for fname in ["modeA_week01.csv", "modeB_week01.csv", "modeC_week01.csv",
                  "ids_ud_week01.csv"]:
        assert (tmp_path / fname).exists()


def test_save_model_column_names(tmp_path):
    cp = _make_cp(n_ud=4, n_min=10, rank=3)
    ids_ud = pd.DataFrame({"user_id": list("abcd"), "day": ["2018-10-22"] * 4})
    save_model(cp, "test", ids_ud, str(tmp_path))
    df = pd.read_csv(tmp_path / "modeA_test.csv")
    assert list(df.columns) == ["final_model1", "final_model2", "final_model3"]
    assert len(df) == 4


def test_save_model_mode_shapes(tmp_path):
    cp = _make_cp(n_ud=5, n_min=10, rank=2)
    ids_ud = pd.DataFrame({"user_id": [f"u{i}" for i in range(5)],
                           "day": ["2018-10-22"] * 5})
    save_model(cp, "s", ids_ud, str(tmp_path))
    assert pd.read_csv(tmp_path / "modeA_s.csv").shape == (5, 2)
    assert pd.read_csv(tmp_path / "modeB_s.csv").shape == (10, 2)
    assert pd.read_csv(tmp_path / "modeC_s.csv").shape == (2, 2)


def test_save_model_ids_content(tmp_path):
    cp = _make_cp(n_ud=2, n_min=10, rank=2)
    ids_ud = pd.DataFrame({"user_id": ["userX", "userY"],
                           "day": ["2018-10-22", "2018-10-23"]})
    save_model(cp, "label", ids_ud, str(tmp_path))
    saved = pd.read_csv(tmp_path / "ids_ud_label.csv")
    assert list(saved["user_id"]) == ["userX", "userY"]
    assert list(saved["day"]) == ["2018-10-22", "2018-10-23"]


# ---------------------------------------------------------------------------
# End-to-end: train_per_week
# ---------------------------------------------------------------------------

def test_train_per_week_end_to_end(tmp_path, monkeypatch):
    down_dir = tmp_path / "input" / "down"
    up_dir = tmp_path / "input" / "up"
    out_dir = tmp_path / "output"
    down_dir.mkdir(parents=True)
    up_dir.mkdir(parents=True)

    uds = [("u1", "2018-10-22"), ("u2", "2018-10-22"),
           ("u3", "2018-10-23"), ("u1", "2018-10-23")]
    _make_week_csv(down_dir / "week01_2018-10-22_2018-10-28.csv", uds, seed=0)
    _make_week_csv(up_dir / "week01_2018-10-22_2018-10-28.csv", uds, seed=1)

    import train_model
    monkeypatch.setattr(train_model, "DOWN_DIR", str(down_dir))
    monkeypatch.setattr(train_model, "UP_DIR", str(up_dir))

    train_per_week(str(out_dir), n_components=2, tol=1e-2)

    week_dir = out_dir / "per_week" / "week01"
    for fname in ["modeA_week01.csv", "modeB_week01.csv",
                  "modeC_week01.csv", "ids_ud_week01.csv"]:
        assert (week_dir / fname).exists(), f"Missing: {fname}"

    assert pd.read_csv(week_dir / "modeA_week01.csv").shape == (4, 2)
    assert pd.read_csv(week_dir / "modeB_week01.csv").shape == (NUM_MINUTES, 2)
    assert pd.read_csv(week_dir / "modeC_week01.csv").shape == (2, 2)
    assert pd.read_csv(week_dir / "ids_ud_week01.csv").shape == (4, 2)

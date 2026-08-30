import calendar
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from preprocess import (
    MAX_CONSEC_NAN,
    MAX_TOTAL_NAN,
    NUM_MINUTES,
    _COL_NAMES,
    _DAILY_PREFIX,
    _compute_max_consec_nan,
    _iso_week_label,
    _iso_week_sunday,
    _load_selection,
    _passes_filter,
    _round_to_minute,
    align,
    align_wan,
    compute_stats,
    filter_series,
    wan_filter,
)


# ---------------------------------------------------------------------------
# _compute_max_consec_nan
# ---------------------------------------------------------------------------

def test_max_consec_nan_no_nans():
    s = pd.Series([1.0, 2.0, 3.0])
    assert _compute_max_consec_nan(s) == 0


def test_max_consec_nan_all_nan():
    s = pd.Series([np.nan] * 5)
    assert _compute_max_consec_nan(s) == 5


def test_max_consec_nan_single_run():
    s = pd.Series([1.0, np.nan, np.nan, np.nan, 2.0])
    assert _compute_max_consec_nan(s) == 3


def test_max_consec_nan_multiple_runs():
    # runs of 2 and 3 — result should be 3
    s = pd.Series([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 2.0])
    assert _compute_max_consec_nan(s) == 3


def test_max_consec_nan_trailing():
    s = pd.Series([1.0, 2.0, np.nan, np.nan])
    assert _compute_max_consec_nan(s) == 2


# ---------------------------------------------------------------------------
# _round_to_minute
# ---------------------------------------------------------------------------

def test_round_to_minute_exact():
    dt = datetime(2018, 10, 22, 9, 30, 0)
    assert _round_to_minute(dt) == datetime(2018, 10, 22, 9, 30, 0)


def test_round_to_minute_rounds_down_at_29s():
    dt = datetime(2018, 10, 22, 9, 30, 29)
    assert _round_to_minute(dt) == datetime(2018, 10, 22, 9, 30, 0)


def test_round_to_minute_rounds_up_at_30s():
    dt = datetime(2018, 10, 22, 9, 30, 30)
    assert _round_to_minute(dt) == datetime(2018, 10, 22, 9, 31, 0)


def test_round_to_minute_midnight_rollover():
    dt = datetime(2018, 10, 22, 23, 59, 45)
    assert _round_to_minute(dt) == datetime(2018, 10, 23, 0, 0, 0)


# ---------------------------------------------------------------------------
# _passes_filter
# ---------------------------------------------------------------------------

def _make_filter_row(consec_nan_positions=None):
    traffic = [1.0] * NUM_MINUTES
    if consec_nan_positions:
        for i in consec_nan_positions:
            traffic[i] = np.nan
    row = pd.Series(["user1", "2018-10-22"] + traffic)
    return row


def test_passes_filter_clean_series():
    row = _make_filter_row()
    assert _passes_filter(row) is True


def test_passes_filter_consec_nan_at_limit():
    # exactly MAX_CONSEC_NAN consecutive NaN → should pass
    row = _make_filter_row(consec_nan_positions=list(range(MAX_CONSEC_NAN)))
    assert _passes_filter(row) is True


def test_passes_filter_consec_nan_over_limit():
    row = _make_filter_row(consec_nan_positions=list(range(MAX_CONSEC_NAN + 1)))
    assert _passes_filter(row) is False


def test_passes_filter_total_nan_at_limit():
    # 719 isolated NaN (alternating positions 0,2,4,...) — each run=1 ≤ 180, total=719 < 720
    positions = list(range(0, MAX_TOTAL_NAN * 2 - 2, 2))  # 0,2,4,...,1436 → 719 NaN
    row = _make_filter_row(consec_nan_positions=positions)
    assert _passes_filter(row) is True


def test_passes_filter_total_nan_at_exact_limit():
    # 720 isolated NaN (alternating positions 0,2,4,...1438) — total=720 → fails
    positions = list(range(0, MAX_TOTAL_NAN * 2, 2))  # 0,2,4,...,1438 → 720 NaN
    row = _make_filter_row(consec_nan_positions=positions)
    assert _passes_filter(row) is False


# ---------------------------------------------------------------------------
# _iso_week_label
# ---------------------------------------------------------------------------

def test_iso_week_label_regular():
    # 2018-10-22 is in ISO week 43 of 2018
    assert _iso_week_label("2018-10-22") == "2018-W43"


def test_iso_week_label_year_boundary():
    # 2018-12-31 falls in ISO week 1 of 2019
    assert _iso_week_label("2018-12-31") == "2019-W01"


def test_iso_week_label_monday():
    # 2018-10-15 is a Monday (start of week 42)
    assert _iso_week_label("2018-10-15") == "2018-W42"


# ---------------------------------------------------------------------------
# _iso_week_sunday
# ---------------------------------------------------------------------------

def test_iso_week_sunday_regular():
    # ISO week 34 of 2026 ends Sunday 2026-08-23
    assert _iso_week_sunday("2026-W34") == datetime(2026, 8, 23).date()


def test_iso_week_sunday_year_boundary():
    # ISO week 1 of 2019 ends Sunday 2019-01-06
    assert _iso_week_sunday("2019-W01") == datetime(2019, 1, 6).date()


# ---------------------------------------------------------------------------
# wan_filter — complete-week guard, stale-file cleanup, interior gaps
# ---------------------------------------------------------------------------

def _write_wan_daily(daily_dir, date_str, hostids=("h1",), minutes=1440):
    """One aligned-WAN daily file: `minutes` full rows per host (enough to
    pass _passes_filter when minutes > MAX_TOTAL_NAN)."""
    rows = []
    for h in hostids:
        for i in range(minutes):
            hh, mm = divmod(i, 60)
            rows.append(
                {
                    "hostid": h,
                    "str_date_hour": f"{date_str} {hh:02d}:{mm:02d}",
                    "bytes_up_dif": 1000.0,
                    "bytes_down_dif": 2000.0,
                }
            )
    pd.DataFrame(rows).to_csv(
        daily_dir / f"wan_metrics_{date_str}.csv", index=False
    )


@pytest.fixture
def wan_filter_setup(tmp_path, monkeypatch):
    import preprocess
    daily_dir = tmp_path / "daily" / "wan_metrics"
    daily_dir.mkdir(parents=True)
    monkeypatch.setattr(preprocess, "WAN_DAILY_DIR", daily_dir)
    monkeypatch.setattr(preprocess, "WAN_INPUT_DIR", tmp_path / "input_wan")
    return daily_dir, tmp_path / "input_wan"


def test_wan_filter_skips_in_progress_week(wan_filter_setup):
    daily_dir, input_dir = wan_filter_setup
    # Full ISO week 34 (Mon 08-17 .. Sun 08-23) + a partial week 35 (Mon, Tue).
    for d in range(17, 24):
        _write_wan_daily(daily_dir, f"2026-08-{d:02d}")
    _write_wan_daily(daily_dir, "2026-08-24")
    _write_wan_daily(daily_dir, "2026-08-25")

    wan_filter()

    down = sorted(p.name for p in (input_dir / "down").glob("*.csv"))
    assert down == ["series_wan_filtered_2026-08-17_2026-08-23.csv"]
    # week 35 (Sunday 2026-08-30) is in-progress -> not written
    assert not list((input_dir / "down").glob("*2026-08-24*"))


def test_wan_filter_removes_stale_output(wan_filter_setup):
    daily_dir, input_dir = wan_filter_setup
    for d in range(17, 24):
        _write_wan_daily(daily_dir, f"2026-08-{d:02d}")
    stale = input_dir / "down"
    stale.mkdir(parents=True)
    (stale / "series_wan_filtered_2019-01-01_2019-01-07.csv").write_text("old")

    wan_filter()

    names = sorted(p.name for p in stale.glob("*.csv"))
    assert names == ["series_wan_filtered_2026-08-17_2026-08-23.csv"]


def test_wan_filter_keeps_past_week_with_interior_gap(wan_filter_setup):
    daily_dir, input_dir = wan_filter_setup
    # ISO week 34, but Wednesday 08-19 is a header-only (empty) daily file,
    # mimicking a real source outage. The week is still calendar-complete.
    for d in range(17, 24):
        if d == 19:
            pd.DataFrame(
                columns=["hostid", "str_date_hour", "bytes_up_dif", "bytes_down_dif"]
            ).to_csv(daily_dir / f"wan_metrics_2026-08-{d:02d}.csv", index=False)
        else:
            _write_wan_daily(daily_dir, f"2026-08-{d:02d}")
    # a later full week so last_data_day is well past week 34
    for d in range(24, 31):
        _write_wan_daily(daily_dir, f"2026-08-{d:02d}")

    wan_filter()

    out = pd.read_csv(
        input_dir / "down" / "series_wan_filtered_2026-08-17_2026-08-23.csv"
    )
    assert sorted(out["day"].unique()) == [
        "2026-08-17", "2026-08-18", "2026-08-20",
        "2026-08-21", "2026-08-22", "2026-08-23",
    ]


# ---------------------------------------------------------------------------
# align_wan — incremental (--start-date) mode
# ---------------------------------------------------------------------------

def _write_wan_raw_utc(raw_dir, date_str, hostids=("aa:bb",)):
    """One raw UTC wan_metrics file: every minute of the UTC day per host."""
    stamps = pd.date_range(f"{date_str} 00:00", f"{date_str} 23:59", freq="1min")
    rows = []
    for h in hostids:
        for t in stamps:
            rows.append(
                {
                    "hostid": h,
                    "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "bytes_up_dif": 100.0,
                    "bytes_down_dif": 200.0,
                    "packets_up_dif": 1.0,
                    "packets_down_dif": 2.0,
                }
            )
    pd.DataFrame(rows).to_csv(
        raw_dir / f"wan_metrics_{date_str}.csv", index=False
    )


@pytest.fixture
def align_wan_setup(tmp_path, monkeypatch):
    import preprocess
    raw_dir = tmp_path / "raw" / "wan_metrics"
    raw_dir.mkdir(parents=True)
    daily_dir = tmp_path / "daily" / "wan_metrics"
    daily_dir.mkdir(parents=True)
    monkeypatch.setattr(preprocess, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "WAN_DAILY_DIR", daily_dir)
    return raw_dir, daily_dir


def test_align_wan_incremental_leaves_earlier_days_untouched(align_wan_setup):
    raw_dir, daily_dir = align_wan_setup
    for d in ("2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13"):
        _write_wan_raw_utc(raw_dir, d)

    # Pre-place a sentinel where an earlier local day's file would be.
    sentinel = daily_dir / "wan_metrics_2026-05-10.csv"
    sentinel.write_text("SENTINEL")

    # Rebuild only local days >= 2026-05-12.
    align_wan(min_samples=720, start_date="2026-05-12")

    # Earlier day: untouched.
    assert sentinel.read_text() == "SENTINEL"
    # Boundary throwaway day (2026-05-11) not written.
    assert not (daily_dir / "wan_metrics_2026-05-11.csv").exists()
    # Target day: rebuilt with real data.
    got = pd.read_csv(daily_dir / "wan_metrics_2026-05-12.csv")
    assert "aa:bb" in set(got["hostid"])
    assert len(got) == NUM_MINUTES  # full local day


def test_align_wan_incremental_matches_full_run_for_kept_days(align_wan_setup):
    raw_dir, daily_dir = align_wan_setup
    for d in ("2026-05-10", "2026-05-11", "2026-05-12", "2026-05-13"):
        _write_wan_raw_utc(raw_dir, d)

    align_wan(min_samples=720)
    full = pd.read_csv(daily_dir / "wan_metrics_2026-05-12.csv").sort_values(
        ["hostid", "str_date_hour"]
    ).reset_index(drop=True)

    (daily_dir / "wan_metrics_2026-05-12.csv").unlink()
    align_wan(min_samples=720, start_date="2026-05-12")
    incr = pd.read_csv(daily_dir / "wan_metrics_2026-05-12.csv").sort_values(
        ["hostid", "str_date_hour"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(full, incr)


def test_align_wan_incremental_no_files_in_range_raises(align_wan_setup):
    raw_dir, _ = align_wan_setup
    _write_wan_raw_utc(raw_dir, "2026-05-10")
    with pytest.raises(FileNotFoundError):
        align_wan(min_samples=720, start_date="2026-06-01")


# ---------------------------------------------------------------------------
# _load_selection
# ---------------------------------------------------------------------------

def test_load_selection_missing_file(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)
    assert _load_selection("down") is None


def test_load_selection_basic(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)
    sel_dir = tmp_path / "down"
    sel_dir.mkdir()
    df = pd.DataFrame({
        "filename": ["series_giga_2018-10-22.csv", "series_giga_2018-10-23.csv"],
        "week": ["week01", "week01"],
        "flag": [1, 0],
    })
    df.to_csv(sel_dir / "selected_allSeries.csv", index=False)
    sel = _load_selection("down")
    assert sel is not None
    assert sel["series_giga_2018-10-22.csv"] == ("week01", 1)
    assert sel["series_giga_2018-10-23.csv"] == ("week01", 0)


def test_load_selection_legacy_column(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)
    sel_dir = tmp_path / "down"
    sel_dir.mkdir()
    df = pd.DataFrame({
        "day_file": ["series_giga_2018-10-22.csv"],
        "week": ["week01"],
        "flag": [1],
    })
    df.to_csv(sel_dir / "selected_allSeries.csv", index=False)
    sel = _load_selection("down")
    assert "series_giga_2018-10-22.csv" in sel


# ---------------------------------------------------------------------------
# align (Stage 2) — synthetic raw CSV
# ---------------------------------------------------------------------------

def _write_raw_csv(path: Path, rows):
    """rows = list of (hostid, timestamp_epoch, traffic_bytes_per_sec)"""
    pd.DataFrame(rows, columns=["hostid", "timestamp", "traffic"]).to_csv(path, index=False)


def test_align_basic(align_setup):
    raw_dir, out_dir = align_setup
    # calendar.timegm gives UTC epoch so pd.to_datetime(unit="s") lands at slot 0
    # regardless of local timezone — needed because post-gap NaN propagation from
    # index 0 would overwrite any measurement at a slot preceded by a gap.
    t0 = calendar.timegm((2018, 10, 22, 0, 0, 0, 0, 0, 0))
    t1 = calendar.timegm((2018, 10, 22, 0, 1, 0, 0, 0, 0))
    _write_raw_csv(raw_dir / "traffic.csv", [
        ("h1", t0, 1000),
        ("h1", t1, 2000),
    ])
    align("down")
    files = list(out_dir.glob("*.csv"))
    assert len(files) == 1
    df = pd.read_csv(files[0])
    assert "user" in df.columns
    assert "h1" in df["user"].values


def test_align_datetime_string_timestamps(align_setup):
    raw_dir, out_dir = align_setup
    _write_raw_csv(raw_dir / "traffic.csv", [
        ("h1", "2018-10-22 00:00:00", 500),
    ])
    align("down")
    assert len(list(out_dir.glob("*.csv"))) == 1


def test_align_dst_duplicate_keeps_first(align_setup):
    """Two identical timestamps → second value is ignored (DST rollback simulation)."""
    raw_dir, out_dir = align_setup
    t = calendar.timegm((2018, 10, 22, 0, 0, 0, 0, 0, 0))
    _write_raw_csv(raw_dir / "traffic.csv", [
        ("h1", t, 1000),
        ("h1", t, 9999),  # duplicate — should be ignored
    ])
    align("down")
    df = pd.read_csv(list(out_dir.glob("*.csv"))[0])
    row = df[df["user"] == "h1"].iloc[0]
    assert row.iloc[2] == pytest.approx(1000 * 60 / 8)  # slot 0 = first value


def test_align_post_gap_nan(align_setup):
    """Minute after a gap is set to NaN (measurement doesn't cover exactly one minute)."""
    raw_dir, out_dir = align_setup
    # Measure at 00:00 and 00:03; 00:01 and 00:02 are missing.
    # Post-gap NaN: 00:01 is NaN → 00:02 set NaN; 00:02 NaN → 00:03 overwritten with NaN.
    t0 = calendar.timegm((2018, 10, 22, 0, 0, 0, 0, 0, 0))
    t3 = calendar.timegm((2018, 10, 22, 0, 3, 0, 0, 0, 0))
    _write_raw_csv(raw_dir / "traffic.csv", [
        ("h1", t0, 100),
        ("h1", t3, 200),
    ])
    align("down")
    df = pd.read_csv(list(out_dir.glob("*.csv"))[0])
    row = df[df["user"] == "h1"].iloc[0]
    base = 2  # first two cols are user, day
    assert not np.isnan(row.iloc[base + 0])   # 00:00 — measurement survives
    assert np.isnan(row.iloc[base + 1])        # 00:01 — no measurement
    assert np.isnan(row.iloc[base + 2])        # 00:02 — post-gap of [1]
    assert np.isnan(row.iloc[base + 3])        # 00:03 — post-gap of [2], overwrites measurement


# ---------------------------------------------------------------------------
# compute_stats (Stage 3)
# ---------------------------------------------------------------------------

def test_compute_stats_output(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)

    daily_dir = tmp_path / "down" / "allSeries"
    daily_dir.mkdir(parents=True)

    traffic = [1.0] * NUM_MINUTES
    traffic[5] = np.nan
    traffic[6] = np.nan
    row = ["h1", "2018-10-22"] + traffic
    df = pd.DataFrame([row], columns=_COL_NAMES)
    df.to_csv(daily_dir / f"{_DAILY_PREFIX}2018-10-22.csv", index=False)

    stats = compute_stats("down")
    assert len(stats) == 1
    assert stats.loc[0, "nan_total"] == 2
    assert stats.loc[0, "max_consec"] == 2


# ---------------------------------------------------------------------------
# filter_series (Stage 4)
# ---------------------------------------------------------------------------

def _write_daily_csv(path: Path, user: str, day: str, traffic: list):
    row = [user, day] + traffic
    pd.DataFrame([row], columns=_COL_NAMES).to_csv(path, index=False)


def test_filter_series_basic(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)
    monkeypatch.setattr(preprocess, "INPUT_DIR", tmp_path / "input")

    daily_dir = tmp_path / "down" / "allSeries"
    daily_dir.mkdir(parents=True)

    _write_daily_csv(
        daily_dir / f"{_DAILY_PREFIX}2018-10-22.csv",
        "h1", "2018-10-22", [1.0] * NUM_MINUTES,
    )

    filter_series("down")

    out_dir = tmp_path / "input" / "down"
    files = list(out_dir.glob("*.csv"))
    assert len(files) == 1
    df = pd.read_csv(files[0])
    assert len(df) == 1
    assert df.iloc[0, 0] == "h1"


def test_filter_series_drops_bad_series(tmp_path, monkeypatch):
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)
    monkeypatch.setattr(preprocess, "INPUT_DIR", tmp_path / "input")

    daily_dir = tmp_path / "down" / "allSeries"
    daily_dir.mkdir(parents=True)

    bad_traffic = [np.nan] * (MAX_CONSEC_NAN + 1) + [1.0] * (NUM_MINUTES - MAX_CONSEC_NAN - 1)
    good_traffic = [1.0] * NUM_MINUTES

    rows = [
        ["bad_user", "2018-10-22"] + bad_traffic,
        ["good_user", "2018-10-22"] + good_traffic,
    ]
    pd.DataFrame(rows, columns=_COL_NAMES).to_csv(
        daily_dir / f"{_DAILY_PREFIX}2018-10-22.csv", index=False
    )

    filter_series("down")

    out_dir = tmp_path / "input" / "down"
    df = pd.read_csv(list(out_dir.glob("*.csv"))[0])
    assert "bad_user" not in df.iloc[:, 0].values
    assert "good_user" in df.iloc[:, 0].values


def test_filter_series_selection_flag0(tmp_path, monkeypatch):
    """Days with flag=0 in selection file are skipped."""
    import preprocess
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path)
    monkeypatch.setattr(preprocess, "INPUT_DIR", tmp_path / "input")

    daily_dir = tmp_path / "down" / "allSeries"
    daily_dir.mkdir(parents=True)

    fname = f"{_DAILY_PREFIX}2018-10-22.csv"
    _write_daily_csv(daily_dir / fname, "h1", "2018-10-22", [1.0] * NUM_MINUTES)

    sel = pd.DataFrame({
        "filename": [fname],
        "week": ["week01"],
        "flag": [0],
    })
    sel.to_csv(tmp_path / "down" / "selected_allSeries.csv", index=False)

    filter_series("down")

    out_dir = tmp_path / "input" / "down"
    assert len(list(out_dir.glob("*.csv"))) == 0

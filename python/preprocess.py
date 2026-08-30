"""
Preprocessing pipeline for PARAFAC traffic model.

Ports SUESTE_OLD preprocessing (mix of Python 2 + R) to a single Python 3 script.

Stages (run in order):
  1   extract     raw measurements (DB) → per-day traffic CSVs
  1b  align_wan   raw wan_metrics CSVs (UTC) → daily 1440-minute long-format
                  CSVs (São Paulo local time), one row per hostid x minute
  1c  wan_filter  aligned WAN daily files → filtered weekly wide CSVs
                  (bytes_down_dif / bytes_up_dif only) for train_model.py
  2   align       raw CSVs → daily 1440-minute series per user
  3   stats       daily series → NaN statistics (info only)
  4   filter      daily series → filtered weekly CSVs for train_model.py
  all             runs stages 2 + 3 + 4

Usage:
  python preprocess.py align_wan   --measure up --min-samples 720
  python preprocess.py align_wan   --measure up --start-date 2026-08-04  # incremental top-up
  python preprocess.py wan_filter  --measure up
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
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import shlex

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

WAN_DAILY_DIR = DATA_DIR / "daily" / "wan_metrics"
WAN_INPUT_DIR = DATA_DIR / "input_wan"
_WAN_TZ = "America/Sao_Paulo"
_WAN_DIF_COLS = ["bytes_up_dif", "bytes_down_dif", "packets_up_dif", "packets_down_dif"]
_WAN_MIN_SAMPLES_DEFAULT = 720  # 50% of 1440 minutes
_WAN_DIRECTION_METRIC = {"down": "bytes_down_dif", "up": "bytes_up_dif"}
# 20 GB in one ~1-minute bucket (~2.67 Gbps sustained) — comfortably above the
# 99.99th percentile of observed legitimate traffic (~2 GB/min); values above
# this are counter-read glitches, not real traffic. See align_wan.
_WAN_MAX_BYTES_PER_MIN = 2e10


def _load_env_file(env_file: str | None, override: bool = True):
    """Load KEY=VALUE pairs from a local file into process environment."""
    if not env_file:
        return

    path = Path(env_file).expanduser()
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()
            if "=" not in line:
                continue

        key, value = line.split("=", 1)
        key = key.strip()
        tokens = shlex.split(value, comments=True, posix=True)
        value = tokens[0] if tokens else ""
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _parse_extract_dates(dates: list[str] | None, start_date: str | None, end_date: str | None) -> list[str]:
    """Return sorted unique YYYY-MM-DD dates from explicit list or inclusive range."""
    if dates:
        parsed: set[str] = set()
        for chunk in dates:
            for token in chunk.split(","):
                token = token.strip()
                if not token:
                    continue
                parsed.add(pd.Timestamp(token).strftime("%Y-%m-%d"))
        if not parsed:
            raise ValueError("No valid dates provided in --dates")
        return sorted(parsed)

    if start_date and end_date:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        if end < start:
            raise ValueError("--end-date must be >= --start-date")
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start=start, end=end, freq="D")]

    raise ValueError(
        "For stage 'extract', provide either --dates YYYY-MM-DD[,YYYY-MM-DD...] "
        "or both --start-date and --end-date."
    )


def _extract_sql(measure: str) -> str:
    """SQL to extract per-minute traffic increments from cumulative byte counters."""
    bytes_col = "bytes_up" if measure == "up" else "bytes_down"
    return f"""
        SELECT
            router_mac,
            timestamp,
            CASE
                WHEN ({bytes_col} - LAG({bytes_col}) OVER (PARTITION BY router_mac ORDER BY timestamp)) < 0 THEN NULL
                ELSE ({bytes_col} - LAG({bytes_col}) OVER (PARTITION BY router_mac ORDER BY timestamp))
            END AS traffic
        FROM wan_metrics
        WHERE timestamp >= :start_ts
          AND timestamp < :end_ts
        ORDER BY router_mac, timestamp;
    """


def _extract_sql_counter(counter_col: str) -> str:
    """SQL for one cumulative counter and its first difference."""
    return f"""
        SELECT
            router_mac,
            timestamp,
            {counter_col} AS counter,
            CASE
                WHEN (
                    {counter_col} - LAG({counter_col}) OVER (
                        PARTITION BY router_mac ORDER BY timestamp
                    )
                ) < 0 THEN NULL
                ELSE (
                    {counter_col} - LAG({counter_col}) OVER (
                        PARTITION BY router_mac ORDER BY timestamp
                    )
                )
            END AS traffic
        FROM wan_metrics
        WHERE timestamp >= :start_ts
          AND timestamp < :end_ts
        ORDER BY router_mac, timestamp;
    """


def _extract_sql_all_metrics() -> str:
    """SQL for all WAN counters and their first differences."""
    return """
        SELECT
            router_mac,
            timestamp,
            bytes_up,
            CASE
                WHEN (bytes_up - LAG(bytes_up) OVER (PARTITION BY router_mac ORDER BY timestamp)) < 0 THEN NULL
                ELSE (bytes_up - LAG(bytes_up) OVER (PARTITION BY router_mac ORDER BY timestamp))
            END AS bytes_up_dif,
            bytes_down,
            CASE
                WHEN (bytes_down - LAG(bytes_down) OVER (PARTITION BY router_mac ORDER BY timestamp)) < 0 THEN NULL
                ELSE (bytes_down - LAG(bytes_down) OVER (PARTITION BY router_mac ORDER BY timestamp))
            END AS bytes_down_dif,
            packets_up,
            CASE
                WHEN (packets_up - LAG(packets_up) OVER (PARTITION BY router_mac ORDER BY timestamp)) < 0 THEN NULL
                ELSE (packets_up - LAG(packets_up) OVER (PARTITION BY router_mac ORDER BY timestamp))
            END AS packets_up_dif,
            packets_down,
            CASE
                WHEN (packets_down - LAG(packets_down) OVER (PARTITION BY router_mac ORDER BY timestamp)) < 0 THEN NULL
                ELSE (packets_down - LAG(packets_down) OVER (PARTITION BY router_mac ORDER BY timestamp))
            END AS packets_down_dif
        FROM wan_metrics
        WHERE timestamp >= :start_ts
          AND timestamp < :end_ts
        ORDER BY router_mac, timestamp;
    """


def _create_engine_with_optional_password(db_url: str, password_env_var: str):
    """Create SQLAlchemy engine, filling password from env var when absent in URL."""
    import sqlalchemy as sa

    parsed = sa.engine.make_url(db_url)
    if parsed.password:
        return sa.create_engine(parsed)

    password = os.getenv(password_env_var)
    if password:
        return sa.create_engine(parsed.set(password=password))

    return sa.create_engine(parsed)


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

def extract(
    measure: str,
    db_url: str | None = None,
    dates: list[str] | None = None,
    counter: str | None = None,
    all_metrics: bool = False,
    db_password_env: str = "WAN_DB_PASSWORD",
):
    """
    Stage 1: extract per-device traffic CSVs from wan_metrics (database).

        Default export (PARAFAC pipeline):
            data/raw/{measure}/<upstream|downstream>_traffic_YYYY-MM-DD.csv
            columns: hostid | timestamp | counter | traffic

        Optional full export (--all-metrics):
            data/raw/wan_metrics/wan_metrics_YYYY-MM-DD.csv
            columns include bytes_* / packets_* and *_dif fields.
    """
    if not db_url:
        raise ValueError("Missing DB URL for extract stage.")
    if not dates:
        raise ValueError("Missing date list for extract stage.")
    extract_from_db(
        measure,
        db_url,
        dates,
        counter=counter,
        all_metrics=all_metrics,
        db_password_env=db_password_env,
    )


def extract_from_db(
        measure: str,
        db_url: str,
        dates: list[str],
        counter: str | None = None,
        all_metrics: bool = False,
        db_password_env: str = "WAN_DB_PASSWORD",
):
    try:
        import sqlalchemy as sa
    except ImportError as exc:
        raise ImportError(
            "Stage 'extract' requires SQLAlchemy. Install with: pip install sqlalchemy"
        ) from exc

    if all_metrics:
        out_dir = RAW_DIR / "wan_metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "wan_metrics"
        query = sa.text(_extract_sql_all_metrics())
    else:
        out_dir = RAW_DIR / measure
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "upstream_traffic" if measure == "up" else "downstream_traffic"
        selected_counter = counter or ("bytes_up" if measure == "up" else "bytes_down")
        query = sa.text(_extract_sql_counter(selected_counter))

    print(f"=== Stage 1: extract [{measure}] — {len(dates)} day(s) ===")

    engine = _create_engine_with_optional_password(db_url, db_password_env)
    with engine.connect() as conn:
        for day in dates:
            start_ts = pd.Timestamp(day)
            end_ts = start_ts + pd.Timedelta(days=1)
            df = pd.read_sql_query(
                query,
                conn,
                params={"start_ts": start_ts.to_pydatetime(), "end_ts": end_ts.to_pydatetime()},
            )

            if "hostid" not in df.columns:
                if "router_mac" in df.columns:
                    df = df.rename(columns={"router_mac": "hostid"})
                elif "mac" in df.columns:
                    df = df.rename(columns={"mac": "hostid"})

            out_path = out_dir / f"{prefix}_{day}.csv"
            df.to_csv(out_path, index=False)
            print(f"  {day}: {len(df)} rows -> {out_path.name}")

    print("Done.\n")


# ---------------------------------------------------------------------------
# Stage 1b: align_wan — port of SUESTE data_processing/filter_module pipeline
# ---------------------------------------------------------------------------

def _align_wan_one_file(raw_csv: Path) -> tuple[pd.DataFrame, int]:
    """
    Read one raw wan_metrics UTC CSV, convert to local time (America/Sao_Paulo),
    truncate to minute, and return mean per (hostid, minute_local) — identical
    to mean_duplicates() in the SUESTE 2021 pipeline.

    Returns:
        df        DataFrame with columns hostid | str_date_hour (YYYY-MM-DD
                  HH:MM, local) | <_dif metrics>
        n_capped  count of counter-diff readings masked by the outlier cap
    """
    df = pd.read_csv(raw_csv)

    # Parse UTC string timestamps and convert to Sao Paulo local time
    # (same logic as filter_module.py: tz_localize UTC → tz_convert Sao_Paulo)
    ts = pd.to_datetime(df["timestamp"])
    ts = ts.dt.tz_localize("UTC").dt.tz_convert(_WAN_TZ)

    df["str_date_hour"] = ts.dt.strftime("%Y-%m-%d %H:%M")  # truncate to minute
    df["str_date"] = ts.dt.strftime("%Y-%m-%d")

    # Keep only _dif columns (rates); drop cumulative counters and raw timestamp
    keep_cols = ["hostid", "str_date", "str_date_hour"] + [
        c for c in _WAN_DIF_COLS if c in df.columns
    ]
    df = df[keep_cols]

    # Outlier cap: a single corrupted counter read (e.g. a momentary bad SNMP
    # poll) produces a spuriously huge NEXT-sample diff — the extract SQL's
    # "diff < 0 -> NULL" only catches the corrupted sample itself, not the
    # inflated diff on the sample right after it. Mask any diff implausibly
    # large for a ~1-minute window (default cap comfortably above observed
    # legitimate traffic, see _WAN_MAX_BYTES_PER_MIN).
    agg_cols = [c for c in _WAN_DIF_COLS if c in df.columns]
    n_capped = 0
    for col in agg_cols:
        outliers = df[col] > _WAN_MAX_BYTES_PER_MIN
        n_capped += int(outliers.sum())
        df.loc[outliers, col] = np.nan

    # mean_duplicates: same as SUESTE — mean per (hostid, minute_local)
    df = (
        df.sort_values(["hostid", "str_date_hour"])
        .groupby(["hostid", "str_date", "str_date_hour"], as_index=False)[agg_cols]
        .mean(numeric_only=True)
    )
    return df, n_capped


def align_wan(min_samples: int = _WAN_MIN_SAMPLES_DEFAULT,
              start_date: str | None = None):
    """
    Stage 1b: align WAN all-metrics raw CSVs to local-time daily files.

    Reads data/raw/wan_metrics/wan_metrics_YYYY-MM-DD.csv (UTC timestamps),
    converts to America/Sao_Paulo, truncates to minute, merges consecutive UTC
    days to produce complete local calendar days, applies a minimum-samples
    filter per (hostid, local_date), and writes:

        data/daily/wan_metrics/wan_metrics_YYYY-MM-DD.csv

    Columns: hostid | str_date_hour | bytes_up_dif | bytes_down_dif |
             packets_up_dif | packets_down_dif

    min_samples: minimum number of valid minute-rows per (hostid, day).
                 Default 720 = 50% of 1440 minutes (same threshold as
                 filter_series MAX_TOTAL_NAN for the classic pipeline).

    start_date: incremental mode. If given (YYYY-MM-DD), only local days
                >= start_date are rebuilt. UTC files before start_date are
                skipped; the first kept UTC file still spills its first ~3h
                into local day (start_date - 1), but that partial day is
                discarded rather than written, so its existing daily file is
                left intact. Local day start_date needs the UTC file for
                start_date + 1 to be complete — make sure it was extracted.
                Use this for cheap top-ups after extending the extraction.
    """
    raw_dir = RAW_DIR / "wan_metrics"
    raw_files = sorted(raw_dir.glob("wan_metrics_*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No wan_metrics CSVs found in {raw_dir}")

    if start_date:
        raw_files = [
            f for f in raw_files
            if f.stem.replace("wan_metrics_", "") >= start_date
        ]
        if not raw_files:
            raise FileNotFoundError(
                f"No wan_metrics CSVs on/after {start_date} in {raw_dir}"
            )

    WAN_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    scope = f", local days >= {start_date}" if start_date else ""
    print(f"=== Stage 1b: align_wan — {len(raw_files)} UTC file(s){scope}, "
          f"min_samples={min_samples} ===")

    # Rolling buffer: accumulate partial local-day DataFrames across UTC files.
    # A local day D is complete after we process UTC file D+1 (because UTC file
    # D+1 contributes the last 3 hours of local day D: 21:00–23:59).
    day_buffer: dict[str, list[pd.DataFrame]] = {}

    def _flush_day(local_date: str):
        """Merge, filter, and write one complete local day."""
        if start_date and local_date < start_date:
            # Throwaway boundary day: the first kept UTC file spills into it,
            # but we never processed the UTC file that owns its bulk. Drop it
            # so the existing daily file is not overwritten with partial data.
            day_buffer.pop(local_date, None)
            return
        parts = day_buffer.pop(local_date, [])
        if not parts:
            print(f"  {local_date}: no data — skipped")
            return

        day_df = pd.concat(parts, ignore_index=True)

        # Re-apply mean_duplicates across merged parts (edge-case safety)
        agg_cols = [c for c in _WAN_DIF_COLS if c in day_df.columns]
        day_df = (
            day_df.groupby(["hostid", "str_date", "str_date_hour"], as_index=False)[agg_cols]
            .mean(numeric_only=True)
        )

        # filter_by_day_samples: keep only hostids with >= min_samples rows
        counts = day_df.groupby("hostid")["str_date_hour"].transform("count")
        before = day_df["hostid"].nunique()
        day_df = day_df[counts >= min_samples].reset_index(drop=True)
        after = day_df["hostid"].nunique()

        out_cols = ["hostid", "str_date_hour"] + agg_cols
        out_path = WAN_DAILY_DIR / f"wan_metrics_{local_date}.csv"
        day_df[out_cols].to_csv(out_path, index=False)
        print(f"  {local_date}: {after} routers ({before - after} dropped, "
              f"{len(day_df)} rows) -> {out_path.name}")

    total_capped = 0
    for i, raw_csv in enumerate(raw_files):
        utc_date = raw_csv.stem.replace("wan_metrics_", "")  # YYYY-MM-DD (UTC)
        df, n_capped = _align_wan_one_file(raw_csv)
        total_capped += n_capped

        # Distribute rows into per-local-date buckets
        for local_date, chunk in df.groupby("str_date"):
            day_buffer.setdefault(local_date, []).append(chunk)

        # After processing UTC file i (date D), local day D-1 is complete:
        # all UTC files that could contribute to it have been processed.
        prev_utc = (pd.Timestamp(utc_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prev_utc in day_buffer:
            _flush_day(prev_utc)

    # Flush remaining days (the last UTC file's local dates)
    for local_date in sorted(day_buffer.keys()):
        _flush_day(local_date)

    print(f"Capped {total_capped} counter-diff outlier reading(s) "
          f"(> {_WAN_MAX_BYTES_PER_MIN:.0e} bytes/min).")
    print("Done.\n")


# ---------------------------------------------------------------------------
# Stage 1c: wan_filter — pivot aligned WAN daily files into weekly wide CSVs
# ---------------------------------------------------------------------------

def _wan_day_to_wide(day_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Pivot one day's long-format WAN rows (hostid x minute) into a wide
    1440-minute row per hostid. Missing minutes become NaN columns."""
    minute = day_df["str_date_hour"].str.slice(-5)
    wide = day_df.assign(minute=minute).pivot_table(
        index="hostid", columns="minute", values=metric_col, aggfunc="mean"
    )
    return wide.reindex(columns=_MINUTE_COLS)


def _iso_week_sunday(week_label: str) -> date:
    """'2026-W34' -> date of that ISO week's Sunday (ISO weekday 7)."""
    iso_year, iso_week = int(week_label[:4]), int(week_label[6:])
    return date.fromisocalendar(iso_year, iso_week, 7)


def wan_filter(min_samples: int = _WAN_MIN_SAMPLES_DEFAULT):
    """
    Stage 1c: pivot data/daily/wan_metrics/ (long format, one row per
    hostid x minute) into weekly wide CSVs matching the schema filter_series
    produces (user | day | 00:00 ... 23:59), so train_model.py can read them
    unmodified via --down-dir/--up-dir data/input_wan/{down,up}.

    Only bytes_down_dif / bytes_up_dif are used — packets are dropped.
    Applies the same quality filter as filter_series (<=180 consecutive NaN,
    <720 total NaN per router per day). Weeks are grouped by ISO calendar week.

    Resumability: this is a full rebuild. data/input_wan/{down,up}/ is cleared
    first, so re-running after extending the extraction never leaves stale
    weekly files behind. Only ISO weeks whose Sunday is <= the last local day
    with data are written — the current, still-in-progress week is skipped so
    it can't land as a short file that later grows under a different name.
    Past weeks with interior gaps (real source outages) are still written with
    whatever days they have.

    min_samples is accepted for symmetry with align_wan but is not applied
    here — the per-day minimum-samples filter already ran in align_wan.
    """
    day_files = sorted(WAN_DAILY_DIR.glob("wan_metrics_*.csv"))
    if not day_files:
        raise FileNotFoundError(f"No daily WAN files found in {WAN_DAILY_DIR}")

    print(f"=== Stage 1c: wan_filter — {len(day_files)} day(s) ===")

    weeks: dict[str, dict[str, list[pd.DataFrame]]] = {"down": {}, "up": {}}
    last_data_day: date | None = None

    for f in day_files:
        date_str = f.stem.replace("wan_metrics_", "")
        raw = pd.read_csv(f)
        week_label = _iso_week_label(date_str)
        if not raw.empty:
            d = date.fromisoformat(date_str)
            last_data_day = d if last_data_day is None else max(last_data_day, d)

        for direction, metric_col in _WAN_DIRECTION_METRIC.items():
            wide = _wan_day_to_wide(raw, metric_col)
            wide = wide.reset_index().rename(columns={"hostid": "user"})
            wide.insert(1, "day", date_str)
            wide = wide[_COL_NAMES]

            before = len(wide)
            passing = wide[wide.apply(_passes_filter, axis=1)].reset_index(drop=True)
            print(f"  {date_str} [{direction}]: {before} -> {len(passing)} routers")
            weeks[direction].setdefault(week_label, []).append(passing)

    for direction in ("down", "up"):
        out_dir = WAN_INPUT_DIR / direction
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("series_wan_filtered_*.csv"):
            stale.unlink()
        for week_label, frames in sorted(weeks[direction].items()):
            if last_data_day is None or _iso_week_sunday(week_label) > last_data_day:
                print(f"  => [{direction}] {week_label}: in-progress week — skipped")
                continue
            week_df = pd.concat(frames, ignore_index=True)
            if week_df.empty:
                continue
            days = sorted(week_df["day"].unique())
            out_path = out_dir / f"series_wan_filtered_{days[0]}_{days[-1]}.csv"
            week_df.to_csv(out_path, index=False)
            print(f"  => [{direction}] {out_path.name} ({len(week_df)} rows, {len(days)} days)")

    print("Done.\n")


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
        choices=["extract", "align_wan", "wan_filter", "align", "stats", "filter", "all"],
    )
    parser.add_argument(
        "--secrets-file",
        default=str(_HERE / ".secrets.env"),
        help="Path to local secrets file (KEY=VALUE), ignored by git.",
    )
    parser.add_argument(
        "--measure", choices=["down", "up", "both"], required=True,
        help="Traffic direction. Use 'both' to process down and up in sequence.",
    )
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="One or more dates (YYYY-MM-DD). You may pass comma-separated values.",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date (YYYY-MM-DD), inclusive. With 'extract', paired with "
             "--end-date to set the query range. With 'align_wan', enables "
             "incremental mode: only local days >= this date are rebuilt.",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date (YYYY-MM-DD), inclusive. Used with --start-date.",
    )
    parser.add_argument(
        "--db-url", default=os.getenv("WAN_DB_URL"),
        help="SQLAlchemy DB URL. If omitted, uses WAN_DB_URL env var.",
    )
    parser.add_argument(
        "--db-password-env", default="WAN_DB_PASSWORD",
        help="Environment variable name for DB password when URL has no password.",
    )
    parser.add_argument(
        "--counter",
        choices=["bytes_up", "bytes_down", "packets_up", "packets_down"],
        default=None,
        help="Counter to difference in extract stage. Defaults by --measure (bytes_up/down).",
    )
    parser.add_argument(
        "--all-metrics", action="store_true",
        help="Extract all WAN counters and *_dif fields in one dataset per day.",
    )
    parser.add_argument(
        "--min-samples", type=int, default=_WAN_MIN_SAMPLES_DEFAULT,
        help="align_wan: minimum minute-rows per router per day (default: 720 = 50%%).",
    )
    args = parser.parse_args()

    _load_env_file(args.secrets_file, override=True)

    if args.db_url is None:
        args.db_url = os.getenv("WAN_DB_URL")

    measures = ["down", "up"] if args.measure == "both" else [args.measure]

    for measure in measures:
        if args.stage == "extract":
            if not args.db_url:
                raise ValueError(
                    "Missing DB URL. Pass --db-url or set WAN_DB_URL environment variable."
                )
            dates = _parse_extract_dates(args.dates, args.start_date, args.end_date)
            extract(
                measure,
                db_url=args.db_url,
                dates=dates,
                counter=args.counter,
                all_metrics=args.all_metrics,
                db_password_env=args.db_password_env,
            )
        elif args.stage == "align_wan":
            align_wan(min_samples=args.min_samples, start_date=args.start_date)
            break  # align_wan is not per-measure
        elif args.stage == "wan_filter":
            wan_filter(min_samples=args.min_samples)
            break  # wan_filter is not per-measure
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

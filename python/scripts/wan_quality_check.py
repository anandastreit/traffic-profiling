from pathlib import Path
import base64
import io
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import pandas as pd  # noqa: E402


_METRICS = ["bytes_up_dif", "bytes_down_dif", "packets_up_dif", "packets_down_dif"]
_METRIC_LABELS = ["bytes_up", "bytes_down", "packets_up", "packets_down"]

# ── thresholds ──────────────────────────────────────────────────────────────
_EMPTY_THRESHOLD = 100        # rows — days below this are considered empty
_NULL_WARN_PCT   = 5.0        # null% above this triggers a yellow/red cell
_NULL_ERR_PCT    = 20.0


def _build_summary(files: list[Path]) -> pd.DataFrame:
    rows = []
    for file_path in files:
        day = file_path.stem.replace("wan_metrics_", "")
        frame = pd.read_csv(file_path, usecols=["hostid", "timestamp"] + _METRICS)
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")

        row = {
            "day": day,
            "rows": len(frame),
            "hosts": frame["hostid"].nunique(dropna=True),
            "timestamp_parse_na": int(timestamps.isna().sum()),
            "timestamp_out_of_day": int(
                (timestamps.dt.strftime("%Y-%m-%d") != day).fillna(False).sum()
            ),
            "dup_hostid_timestamp": int(frame[["hostid", "timestamp"]].duplicated().sum()),
        }
        for metric in _METRICS:
            series = frame[metric]
            row[f"{metric}_null_pct"] = float(series.isna().mean() * 100)
            row[f"{metric}_neg_count"] = int((series.dropna() < 0).sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def _build_align_summary(logs_dir: Path, raw_summary: pd.DataFrame) -> pd.DataFrame:
    """Parse align_wan.log to get kept/dropped counts per day — fast, no CSV reading."""
    log_path = logs_dir / "align_wan.log"
    if not log_path.exists():
        return pd.DataFrame()

    import re
    rows = []
    # Line format: "  2026-07-01: 4757 routers (75 dropped, 6784155 rows) -> ..."
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}):\s+(\d+) routers \((\d+) dropped")
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            rows.append({
                "day": m.group(1),
                "hosts_aligned": int(m.group(2)),
                "dropped": int(m.group(3)),
            })

    if not rows:
        return pd.DataFrame()

    align_df = pd.DataFrame(rows)
    merged = raw_summary[["day", "hosts"]].merge(align_df, on="day", how="left")
    merged["hosts_aligned"] = merged["hosts_aligned"].fillna(0).astype(int)
    merged["dropped"] = merged["dropped"].fillna(0).astype(int)
    return merged


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_routers_per_day(summary: pd.DataFrame) -> str:
    non_empty = summary[summary["rows"] >= _EMPTY_THRESHOLD].copy()
    dates = pd.to_datetime(non_empty["day"])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(dates, non_empty["hosts"], alpha=0.15, color="#0d6efd")
    ax.plot(dates, non_empty["hosts"], color="#0d6efd", lw=1.5, label="raw (UTC day)")
    ax.set_title("Routers per day (raw UTC files)", fontsize=12)
    ax.set_ylabel("# routers")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_dropped_per_day(align_df: pd.DataFrame) -> str:
    if align_df.empty:
        return ""
    non_empty = align_df[align_df["hosts"] >= _EMPTY_THRESHOLD].copy()
    dates = pd.to_datetime(non_empty["day"])

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(dates, non_empty["dropped"], color="#dc3545", alpha=0.7, width=0.8, label="dropped by min_samples filter")
    ax.bar(dates, non_empty["hosts_aligned"], bottom=non_empty["dropped"], color="#198754",
           alpha=0.7, width=0.8, label="kept")
    ax.set_title("Routers kept vs dropped by align_wan filter (min_samples=720)", fontsize=12)
    ax.set_ylabel("# routers")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_null_pct(summary: pd.DataFrame) -> str:
    non_empty = summary[summary["rows"] >= _EMPTY_THRESHOLD].copy()
    colors = ["#0d6efd", "#fd7e14", "#198754", "#dc3545"]

    avgs = [non_empty[f"{m}_null_pct"].mean() for m in _METRICS]
    maxs = [non_empty[f"{m}_null_pct"].max() for m in _METRICS]

    x = range(len(_METRIC_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_avg = ax.bar([i - width / 2 for i in x], avgs, width,
                      label="avg null%", color=colors, alpha=0.85)
    bars_max = ax.bar([i + width / 2 for i in x], maxs, width,
                      label="max null% (worst day)", color=colors, alpha=0.4,
                      edgecolor=[c for c in colors], linewidth=1.2)

    ax.set_title("Null % per metric (_dif columns)", fontsize=12)
    ax.set_ylabel("null %")
    ax.set_xticks(list(x))
    ax.set_xticklabels(_METRIC_LABELS, rotation=15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars_avg:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_max:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_null_over_time(summary: pd.DataFrame) -> str:
    non_empty = summary[summary["rows"] >= _EMPTY_THRESHOLD].copy()
    dates = pd.to_datetime(non_empty["day"])
    colors = ["#0d6efd", "#fd7e14", "#198754", "#dc3545"]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    for metric, label, color in zip(_METRICS, _METRIC_LABELS, colors):
        ax.plot(dates, non_empty[f"{metric}_null_pct"], label=label,
                color=color, lw=1.0, alpha=0.75, marker="o", markersize=2)

    ax.set_title("Null % per day (per metric)", fontsize=12)
    ax.set_ylabel("null %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_rows_per_day(summary: pd.DataFrame) -> str:
    non_empty = summary[summary["rows"] >= _EMPTY_THRESHOLD].copy()
    dates = pd.to_datetime(non_empty["day"])

    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.fill_between(dates, non_empty["rows"] / 1e6, alpha=0.15, color="#6f42c1")
    ax.plot(dates, non_empty["rows"] / 1e6, color="#6f42c1", lw=1.5)
    ax.set_title("Rows per day (millions)", fontsize=12)
    ax.set_ylabel("rows (M)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _img_tag(b64: str) -> str:
    if not b64:
        return "<p><em>Chart not available</em></p>"
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:960px">'


def _cell_style(value: float, warn: float, err: float) -> str:
    if value >= err:
        return "background:#f8d7da"
    if value >= warn:
        return "background:#fff3cd"
    return ""


def _build_html(summary: pd.DataFrame, align_df: pd.DataFrame, generated_at: str) -> str:
    empty_days = summary[summary["rows"] < _EMPTY_THRESHOLD]["day"].tolist()
    non_empty = summary[summary["rows"] >= _EMPTY_THRESHOLD]

    # charts
    img_routers      = _img_tag(_chart_routers_per_day(summary))
    img_dropped      = _img_tag(_chart_dropped_per_day(align_df))
    img_null_summary = _img_tag(_chart_null_pct(summary))
    img_null_time    = _img_tag(_chart_null_over_time(summary))
    img_rows         = _img_tag(_chart_rows_per_day(summary))

    # ── metric summary table ─────────────────────────────────────────────────
    metric_rows = []
    for m, lbl in zip(_METRICS, _METRIC_LABELS):
        col = f"{m}_null_pct"
        avg = non_empty[col].mean()
        mx  = non_empty[col].max()
        mx_day = non_empty.loc[non_empty[col].idxmax(), "day"] if len(non_empty) else "—"
        neg = int(non_empty[f"{m}_neg_count"].sum())
        s_avg = _cell_style(avg, _NULL_WARN_PCT, _NULL_ERR_PCT)
        s_max = _cell_style(mx,  _NULL_WARN_PCT, _NULL_ERR_PCT)
        metric_rows.append(
            f"<tr><td>{lbl}</td>"
            f'<td style="{s_avg}">{avg:.4f}%</td>'
            f'<td style="{s_max}">{mx:.4f}%</td>'
            f"<td>{mx_day}</td><td>{neg:,}</td></tr>"
        )

    # ── per-day table ────────────────────────────────────────────────────────
    header_cells = "".join(
        f"<th>{c}</th>"
        for c in ["day", "rows", "hosts (raw)", "ts_parse_na", "ts_out_of_day", "dups"]
        + [f"{lbl}_null%" for lbl in _METRIC_LABELS]
        + [f"{lbl}_neg" for lbl in _METRIC_LABELS]
    )
    body_rows = []
    for _, r in summary.iterrows():
        is_empty = r["rows"] < _EMPTY_THRESHOLD
        row_style = "background:#e9ecef" if is_empty else ""
        cells = [
            f'<td style="{row_style}">{r["day"]}</td>',
            f'<td style="{row_style}">{int(r["rows"]):,}</td>',
            f'<td style="{row_style}">{int(r["hosts"]):,}</td>',
            f'<td style="{row_style}">{int(r["timestamp_parse_na"])}</td>',
            f'<td style="{row_style}">{int(r["timestamp_out_of_day"])}</td>',
            f'<td style="{row_style}">{int(r["dup_hostid_timestamp"])}</td>',
        ]
        for m in _METRICS:
            v = r[f"{m}_null_pct"]
            s = _cell_style(v, _NULL_WARN_PCT, _NULL_ERR_PCT) or row_style
            cells.append(f'<td style="{s}">{v:.2f}%</td>')
        for m in _METRICS:
            cells.append(f'<td style="{row_style}">{int(r[f"{m}_neg_count"])}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    empty_list_html = (
        "<p>None ✅</p>"
        if not empty_days
        else "<ul>" + "".join(f"<li>{d}</li>" for d in empty_days) + "</ul>"
    )

    n_files    = len(summary)
    n_empty    = len(empty_days)
    n_ok       = n_files - n_empty
    rows_total = int(summary["rows"].sum())

    filters_html = textwrap.dedent("""
    <ul>
      <li><strong>SQL (extract stage):</strong> negative deltas (<code>value_diff &lt; 0</code>)
          set to NULL — counter resets (router reboot) are discarded.</li>
      <li><strong>align_wan stage — min_samples filter:</strong> routers with fewer than
          <strong>720 minute-rows per local day</strong> (&lt; 50% of 1440 min) are dropped.
          Equivalent to the <code>filter_by_day_samples</code> used in the SUESTE 2021 pipeline.</li>
      <li><strong>Timezone:</strong> raw UTC timestamps converted to
          <code>America/Sao_Paulo</code> before grouping into calendar days
          (same as <code>filter_module.py</code> — SUESTE 2021).</li>
      <li><strong>Duplicate timestamps:</strong> multiple readings for the same
          (router, UTC minute) are averaged (<code>mean_duplicates</code>).</li>
    </ul>
    """).strip()

    # Legend: only show warn/error swatches if at least one cell would be colored
    has_warn = any(
        r[f"{m}_null_pct"] >= _NULL_WARN_PCT
        for _, r in non_empty.iterrows()
        for m in _METRICS
    ) if len(non_empty) else False
    max_null_seen = max(non_empty[f"{m}_null_pct"].max() for m in _METRICS) if len(non_empty) else 0.0
    if has_warn:
        legend_html = (
            f'<p class="legend">'
            f'<span class="sq empty-bg"></span>empty day (&lt; {_EMPTY_THRESHOLD} rows) &nbsp;'
            f'<span class="sq warn"></span>null ≥ {_NULL_WARN_PCT}% &nbsp;'
            f'<span class="sq err"></span>null ≥ {_NULL_ERR_PCT}%'
            f'</p>'
        )
    else:
        legend_html = (
            f'<p class="legend">'
            f'<span class="sq empty-bg"></span>empty day (&lt; {_EMPTY_THRESHOLD} rows) &nbsp;'
            f'&mdash; no null threshold exceeded (max observed: {max_null_seen:.2f}%)'
            f'</p>'
        )

    return textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>WAN Metrics Quality Report</title>
      <style>
        body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 32px 40px;
               font-size: 13px; color: #212529; background: #fff; }}
        h1 {{ color: #212529; margin-bottom: 4px; font-size: 22px; }}
        h2 {{ color: #343a40; font-size: 16px; margin-top: 32px; border-bottom: 2px solid #dee2e6;
              padding-bottom: 6px; }}
        h3 {{ font-size: 13px; color: #495057; margin: 18px 0 6px; }}
        p  {{ margin: 4px 0 10px; color: #495057; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12px; }}
        th, td {{ border: 1px solid #dee2e6; padding: 5px 9px; text-align: right; white-space: nowrap; }}
        th {{ background: #343a40; color: #fff; text-align: center;
              position: sticky; top: 0; z-index: 1; }}
        td:first-child {{ text-align: left; font-weight: 500; }}
        tbody tr:hover {{ background: #f8f9fa !important; }}
        .banner {{ display:flex; gap:14px; flex-wrap:wrap; margin: 18px 0 24px; }}
        .kpi {{ background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px;
                padding:14px 22px; min-width:110px; text-align:center; }}
        .kpi .val {{ font-size:26px; font-weight:700; color:#0d6efd; }}
        .kpi .lbl {{ font-size:11px; color:#6c757d; margin-top:3px; }}
        .legend {{ font-size:11px; margin: 8px 0 4px; color: #6c757d; }}
        .sq {{ display:inline-block; width:12px; height:12px; margin-right:4px;
               border:1px solid #ccc; vertical-align:middle; }}
        .warn {{ background:#fff3cd; }} .err {{ background:#f8d7da; }}
        .empty-bg {{ background:#e9ecef; }}
        .scroll {{ overflow-x:auto; border-radius:4px; }}
        img {{ border-radius:6px; border:1px solid #dee2e6; margin: 8px 0; }}
        code {{ background:#f8f9fa; padding:1px 5px; border-radius:3px; font-size:12px; }}
        ul {{ padding-left:22px; line-height:1.7; }}
      </style>
    </head>
    <body>
      <h1>WAN Metrics Quality Report</h1>
      <p>Generated: <strong>{generated_at}</strong> &nbsp;|&nbsp;
         Source: <code>data/raw/wan_metrics/</code> &nbsp;|&nbsp;
         Aligned: <code>data/daily/wan_metrics/</code></p>

      <div class="banner">
        <div class="kpi"><div class="val">{n_files}</div><div class="lbl">total days</div></div>
        <div class="kpi"><div class="val">{n_ok}</div><div class="lbl">days with data</div></div>
        <div class="kpi"><div class="val">{n_empty}</div><div class="lbl">empty days</div></div>
        <div class="kpi"><div class="val">{rows_total:,}</div><div class="lbl">total raw rows</div></div>
        <div class="kpi"><div class="val">{int(non_empty["hosts"].max()) if len(non_empty) else 0:,}</div>
             <div class="lbl">peak routers/day</div></div>
      </div>

      <h2>Preprocessing filters</h2>
      {filters_html}

      <h2>Routers per day</h2>
      <p>Count of unique routers in raw UTC files (before align_wan filter).</p>
      {img_routers}

      <h2>Routers kept vs dropped after align_wan</h2>
      <p>Routers dropped = fewer than 720 valid minute-rows in the local calendar day (SP timezone).</p>
      {img_dropped}

      <h2>Rows per day</h2>
      {img_rows}

      <h2>Null % per metric — summary</h2>
      <p>Average (solid bar) and worst-day maximum (faded bar) across all non-empty days.
         First row of each router/day is always null (no previous value to diff against).</p>
      {img_null_summary}

      <h2>Null % per metric — per day</h2>
      <p>Daily null% for each <code>*_dif</code> column. Spikes indicate days with collection gaps.</p>
      {img_null_time}

      <h2>Metric summary (non-empty days)</h2>
      <table>
        <tr><th>Metric</th><th>Avg null%</th><th>Max null%</th><th>Max null day</th><th>Neg values</th></tr>
        {''.join(metric_rows)}
      </table>

      <h2>Empty days ({n_empty})</h2>
      {empty_list_html}
      <p class="legend">
        <span class="sq warn"></span>null ≥ {_NULL_WARN_PCT}% &nbsp;
        <span class="sq err"></span>null ≥ {_NULL_ERR_PCT}% &nbsp;
        <span class="sq empty-bg"></span>empty day (&lt; {_EMPTY_THRESHOLD} rows)
      </p>

      <h2>Per-day detail</h2>
      {legend_html}
      <div class="scroll">
        <table>
          <tr>{header_cells}</tr>
          {''.join(body_rows)}
        </table>
      </div>
    </body>
    </html>
    """).strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true",
                        help="Reuse existing wan_metrics_quality_summary.csv instead of re-reading raw files.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "wan_metrics_quality_summary.csv"

    if args.use_cache and summary_path.exists():
        print(f"Loading cached summary from {summary_path}")
        summary = pd.read_csv(summary_path)
    else:
        base = root / "data/raw/wan_metrics"
        files = sorted(base.glob("wan_metrics_*.csv"))
        if not files:
            raise FileNotFoundError(f"No daily files found under {base}")
        print(f"Processing {len(files)} files…")
        summary = _build_summary(files)
        summary.to_csv(summary_path, index=False)

    align_df = _build_align_summary(logs_dir, summary)

    from datetime import datetime
    html = _build_html(summary, align_df, datetime.now().strftime("%Y-%m-%d %H:%M"))
    html_path = logs_dir / "wan_metrics_quality_report.html"
    html_path.write_text(html, encoding="utf-8")

    # ── stdout summary ───────────────────────────────────────────────────────
    non_empty = summary[summary["rows"] >= _EMPTY_THRESHOLD]
    print(f"files={len(summary)}")
    print(f"rows_total={int(summary.rows.sum())}")
    print(f"rows_min={int(summary.rows.min())} day={summary.loc[summary.rows.idxmin(), 'day']}")
    print(f"rows_max={int(summary.rows.max())} day={summary.loc[summary.rows.idxmax(), 'day']}")
    print(f"hosts_min={int(non_empty.hosts.min())} day={non_empty.loc[non_empty.hosts.idxmin(), 'day']}")
    print(f"hosts_max={int(non_empty.hosts.max())} day={non_empty.loc[non_empty.hosts.idxmax(), 'day']}")
    print(f"ts_parse_na_total={int(summary.timestamp_parse_na.sum())}")
    print(f"ts_out_of_day_total={int(summary.timestamp_out_of_day.sum())}")
    print(f"dup_total={int(summary.dup_hostid_timestamp.sum())}")
    for metric in _METRICS:
        null_mean = non_empty[f"{metric}_null_pct"].mean()
        max_idx   = non_empty[f"{metric}_null_pct"].idxmax()
        null_max  = non_empty.loc[max_idx, f"{metric}_null_pct"]
        max_day   = non_empty.loc[max_idx, "day"]
        neg_total = int(non_empty[f"{metric}_neg_count"].sum())
        print(f"{metric}: null_mean={null_mean:.4f}% null_max={null_max:.4f}% day={max_day} neg_total={neg_total}")

    print(f"summary_csv={summary_path}")
    print(f"report_html={html_path}")


if __name__ == "__main__":
    main()

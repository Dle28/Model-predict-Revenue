from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def save_recovery_anchor_plot(recovery_diag: pd.DataFrame, path: Path) -> None:
    if recovery_diag.empty:
        return
    frame = recovery_diag.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])

    def none_if_nan(value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        return round(float(value), 2)

    records = []
    for row in frame.itertuples(index=False):
        record = {
            "weekId": getattr(row, "week_id"),
            "weekStart": pd.Timestamp(getattr(row, "week_start")).strftime("%Y-%m-%d"),
        }
        for target in ["revenue", "cogs"]:
            mapping = {
                f"{target}Base": f"{target}_w_base",
                f"{target}Final": f"{target}_w_pred",
                f"{target}Baseline": f"{target}_w_pre_covid_baseline_same_week",
                f"{target}Anchor": f"{target}_w_recovery_anchor",
                f"{target}Progress": f"{target}_w_recovery_progress",
            }
            for key, col in mapping.items():
                record[key] = none_if_nan(getattr(row, col)) if hasattr(row, col) else None
        records.append(record)

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Recovery Anchor Diagnostic</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --ink: #17202a;
      --muted: #667085;
      --grid: #e5e7eb;
      --panel: #fff;
      --forecast: #2563eb;
      --baseline: #dc2626;
      --anchor: #059669;
      --final: #7c3aed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1240px; margin: 0 auto; padding: 24px; }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }
    h1 { margin: 0 0 4px; font-size: 24px; letter-spacing: 0; }
    .subtle { color: var(--muted); }
    .controls {
      display: inline-flex;
      padding: 4px;
      border: 1px solid var(--grid);
      border-radius: 8px;
      background: var(--panel);
      gap: 4px;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: var(--muted);
      font: inherit;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.active { background: #111827; color: #fff; }
    .chart-panel {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 12px;
    }
    canvas { display: block; width: 100%; height: 560px; }
    .legend {
      display: flex;
      align-items: center;
      gap: 18px;
      margin: 10px 4px 0;
      color: var(--muted);
      flex-wrap: wrap;
    }
    .legend-item { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 22px; height: 3px; border-radius: 999px; display: inline-block; }
    .forecast { background: var(--forecast); }
    .final { background: var(--final); }
    .baseline { background: var(--baseline); }
    .anchor { background: var(--anchor); }
    .tooltip {
      position: absolute;
      pointer-events: none;
      background: rgba(17, 24, 39, 0.94);
      color: #fff;
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 12px;
      min-width: 210px;
      transform: translate(-50%, -110%);
      display: none;
      z-index: 2;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Recovery Anchor Diagnostic</h1>
        <div class="subtle">Weekly forecast versus pre-COVID baseline and recovery anchor</div>
      </div>
      <div class="controls" aria-label="Metric selector">
        <button id="revenueBtn" class="active" type="button">Revenue</button>
        <button id="cogsBtn" type="button">COGS</button>
      </div>
    </header>
    <section class="chart-panel">
      <canvas id="chart"></canvas>
      <div class="tooltip" id="tooltip"></div>
      <div class="legend">
        <span class="legend-item"><span class="swatch forecast"></span>LV1 base</span>
        <span class="legend-item"><span class="swatch final"></span>Final weekly sum</span>
        <span class="legend-item"><span class="swatch baseline"></span>Pre-COVID baseline</span>
        <span class="legend-item"><span class="swatch anchor"></span>Recovery anchor</span>
      </div>
    </section>
  </main>
  <script>
    const DATA = __DATA_JSON__;
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const revenueBtn = document.getElementById("revenueBtn");
    const cogsBtn = document.getElementById("cogsBtn");
    let metric = "revenue";
    let hoverIndex = null;

    function money(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return "VND " + Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    function keys() {
      if (metric === "revenue") {
        return { base: "revenueBase", final: "revenueFinal", baseline: "revenueBaseline", anchor: "revenueAnchor", progress: "revenueProgress" };
      }
      return { base: "cogsBase", final: "cogsFinal", baseline: "cogsBaseline", anchor: "cogsAnchor", progress: "cogsProgress" };
    }
    function bounds(k) {
      const vals = [];
      DATA.forEach(row => [k.base, k.final, k.baseline, k.anchor].forEach(key => {
        const v = row[key];
        if (v !== null && Number.isFinite(v)) vals.push(v);
      }));
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const pad = (max - min) * 0.08 || max * 0.08 || 1;
      return { min: Math.max(0, min - pad), max: max + pad };
    }
    function drawLine(points, color, width, dash = []) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let open = false;
      points.forEach(p => {
        if (p.y === null) {
          open = false;
          return;
        }
        if (!open) {
          ctx.moveTo(p.x, p.y);
          open = true;
        } else {
          ctx.lineTo(p.x, p.y);
        }
      });
      ctx.stroke();
      ctx.restore();
    }
    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }
    function draw() {
      const isRevenue = metric === "revenue";
      revenueBtn.classList.toggle("active", isRevenue);
      cogsBtn.classList.toggle("active", !isRevenue);
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const pad = { left: 74, right: 24, top: 20, bottom: 42 };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;
      const k = keys();
      const b = bounds(k);
      const x = i => pad.left + (DATA.length <= 1 ? 0 : i * plotW / (DATA.length - 1));
      const y = v => pad.top + (b.max - v) * plotH / (b.max - b.min);

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#667085";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 5; i++) {
        const val = b.min + (b.max - b.min) * i / 5;
        const yy = y(val);
        ctx.beginPath();
        ctx.moveTo(pad.left, yy);
        ctx.lineTo(w - pad.right, yy);
        ctx.stroke();
        ctx.fillText(money(val), pad.left - 8, yy);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      [0, Math.floor(DATA.length * 0.25), Math.floor(DATA.length * 0.5), Math.floor(DATA.length * 0.75), DATA.length - 1].forEach(i => {
        ctx.fillText(DATA[i].weekStart, x(i), h - pad.bottom + 16);
      });

      drawLine(DATA.map((row, i) => ({ x: x(i), y: row[k.baseline] === null ? null : y(row[k.baseline]) })), "#dc2626", 2, [6, 4]);
      drawLine(DATA.map((row, i) => ({ x: x(i), y: row[k.anchor] === null ? null : y(row[k.anchor]) })), "#059669", 2.5);
      drawLine(DATA.map((row, i) => ({ x: x(i), y: row[k.final] === null ? null : y(row[k.final]) })), "#7c3aed", 2);
      drawLine(DATA.map((row, i) => ({ x: x(i), y: row[k.base] === null ? null : y(row[k.base]) })), "#2563eb", 2.5);

      if (hoverIndex !== null) {
        const row = DATA[hoverIndex];
        const xx = x(hoverIndex);
        const anchorY = row[k.base] !== null ? y(row[k.base]) : pad.top;
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xx, pad.top);
        ctx.lineTo(xx, h - pad.bottom);
        ctx.stroke();
        tooltip.style.display = "block";
        tooltip.style.left = `${Math.min(Math.max(xx, 120), w - 120)}px`;
        tooltip.style.top = `${Math.max(anchorY, 90)}px`;
        tooltip.innerHTML = `
          <strong>${row.weekId}</strong><br>
          LV1 base: ${money(row[k.base])}<br>
          Final sum: ${money(row[k.final])}<br>
          Pre-COVID baseline: ${money(row[k.baseline])}<br>
          Recovery anchor: ${money(row[k.anchor])}<br>
          Progress: ${row[k.progress] === null ? "-" : (Number(row[k.progress]) * 100).toFixed(1) + "%"}
        `;
      } else {
        tooltip.style.display = "none";
      }
    }
    canvas.addEventListener("mousemove", event => {
      const rect = canvas.getBoundingClientRect();
      const xPos = event.clientX - rect.left;
      const plotW = rect.width - 74 - 24;
      const idx = Math.round((xPos - 74) * (DATA.length - 1) / plotW);
      hoverIndex = Math.min(Math.max(idx, 0), DATA.length - 1);
      draw();
    });
    canvas.addEventListener("mouseleave", () => {
      hoverIndex = null;
      draw();
    });
    revenueBtn.addEventListener("click", () => { metric = "revenue"; draw(); });
    cogsBtn.addEventListener("click", () => { metric = "cogs"; draw(); });
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""
    html = html.replace("__DATA_JSON__", json.dumps(records, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"saved {path} ({len(records)} plotted weeks)")


def save_submission_plot(
    submission: pd.DataFrame,
    intervals: pd.DataFrame,
    sales_daily: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    hist = sales_daily.copy()
    hist["Date"] = pd.to_datetime(hist["date"])
    hist = hist.rename(columns={"revenue": "ActualRevenue", "cogs": "ActualCOGS"})

    forecast = intervals.copy() if len(intervals) else submission.copy()
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    if "Revenue_p10" not in forecast.columns:
        forecast["Revenue_p10"] = forecast["Revenue"] * 0.85
        forecast["Revenue_p90"] = forecast["Revenue"] * 1.15
        forecast["Revenue_p05"] = forecast["Revenue"] * 0.78
        forecast["Revenue_p95"] = forecast["Revenue"] * 1.25
        forecast["COGS_p10"] = forecast["COGS"] * 0.85
        forecast["COGS_p90"] = forecast["COGS"] * 1.15
        forecast["COGS_p05"] = forecast["COGS"] * 0.78
        forecast["COGS_p95"] = forecast["COGS"] * 1.25

    forecast_start = forecast["Date"].min()
    hist = hist[hist["Date"].ge(forecast_start - pd.Timedelta(days=365))]

    hist_keep = hist[["Date", "ActualRevenue", "ActualCOGS"]]
    forecast_keep = forecast[
        [
            "Date",
            "Revenue",
            "COGS",
            "Revenue_p10",
            "Revenue_p90",
            "Revenue_p05",
            "Revenue_p95",
            "COGS_p10",
            "COGS_p90",
            "COGS_p05",
            "COGS_p95",
        ]
    ]
    plot_df = hist_keep.merge(forecast_keep, on="Date", how="outer").sort_values("Date")
    plot_df["Date"] = plot_df["Date"].dt.strftime("%Y-%m-%d")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    def none_if_nan(value: Any) -> Optional[float]:
        if pd.isna(value):
            return None
        return round(float(value), 2)

    records = []
    for row in plot_df.itertuples(index=False):
        records.append(
            {
                "date": row.Date,
                "actualRevenue": none_if_nan(row.ActualRevenue),
                "actualCOGS": none_if_nan(row.ActualCOGS),
                "revenue": none_if_nan(row.Revenue),
                "cogs": none_if_nan(row.COGS),
                "revenueP10": none_if_nan(row.Revenue_p10),
                "revenueP90": none_if_nan(row.Revenue_p90),
                "revenueP05": none_if_nan(row.Revenue_p05),
                "revenueP95": none_if_nan(row.Revenue_p95),
                "cogsP10": none_if_nan(row.COGS_p10),
                "cogsP90": none_if_nan(row.COGS_p90),
                "cogsP05": none_if_nan(row.COGS_p05),
                "cogsP95": none_if_nan(row.COGS_p95),
            }
        )

    metric_row = metrics.iloc[0].to_dict() if len(metrics) else {}
    summary = {
        "forecastStart": forecast["Date"].min().strftime("%Y-%m-%d"),
        "forecastEnd": forecast["Date"].max().strftime("%Y-%m-%d"),
        "forecastDays": int(len(forecast)),
        "totalRevenue": round(float(forecast["Revenue"].sum()), 2),
        "totalCOGS": round(float(forecast["COGS"].sum()), 2),
        "weeklyRevenueWape": none_if_nan(metric_row.get("weekly_revenue_wape")),
        "weeklyRevenueR2": none_if_nan(metric_row.get("weekly_revenue_r2")),
        "dailyRevenueWape": none_if_nan(metric_row.get("daily_revenue_wape")),
        "dailyRevenueR2": none_if_nan(metric_row.get("daily_revenue_r2")),
        "weeklyCogsWape": none_if_nan(metric_row.get("weekly_cogs_wape")),
        "weeklyCogsR2": none_if_nan(metric_row.get("weekly_cogs_r2")),
        "dailyCogsWape": none_if_nan(metric_row.get("daily_cogs_wape")),
        "dailyCogsR2": none_if_nan(metric_row.get("daily_cogs_r2")),
    }

    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Submission Forecast Plot</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --grid: #e5e7eb;
      --actual: #1f2937;
      --forecast: #2563eb;
      --band: rgba(37, 99, 235, 0.16);
      --band-wide: rgba(37, 99, 235, 0.08);
      --accent: #059669;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 24px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .subtle { color: var(--muted); }
    .controls {
      display: inline-flex;
      padding: 4px;
      border: 1px solid var(--grid);
      border-radius: 8px;
      background: var(--panel);
      gap: 4px;
    }
    button {
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: var(--muted);
      font: inherit;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.active {
      background: #111827;
      color: #fff;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat {
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .value {
      font-size: 18px;
      font-weight: 680;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .chart-panel {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 12px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 560px;
    }
    .legend {
      display: flex;
      align-items: center;
      gap: 18px;
      margin: 10px 4px 0;
      color: var(--muted);
      flex-wrap: wrap;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }
    .swatch {
      width: 22px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
      background: var(--forecast);
    }
    .swatch.actual { background: var(--actual); }
    .swatch.band { height: 10px; background: var(--band); border: 1px solid rgba(37,99,235,0.2); }
    .tooltip {
      position: absolute;
      pointer-events: none;
      background: rgba(17, 24, 39, 0.94);
      color: white;
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 12px;
      min-width: 176px;
      transform: translate(-50%, -110%);
      display: none;
      z-index: 2;
    }
    .note {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 760px) {
      main { padding: 16px; }
      header { flex-direction: column; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      canvas { height: 430px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Submission Forecast</h1>
        <div class="subtle" id="horizon"></div>
      </div>
      <div class="controls" aria-label="Metric selector">
        <button id="revenueBtn" class="active" type="button">Revenue</button>
        <button id="cogsBtn" type="button">COGS</button>
      </div>
    </header>

    <section class="stats">
      <div class="stat"><div class="label">Forecast total</div><div class="value" id="totalValue"></div></div>
      <div class="stat"><div class="label">Weekly WAPE / R2</div><div class="value" id="weeklyMetric"></div></div>
      <div class="stat"><div class="label">Daily WAPE / R2</div><div class="value" id="dailyMetric"></div></div>
      <div class="stat"><div class="label">Forecast days</div><div class="value" id="daysValue"></div></div>
    </section>

    <section class="chart-panel">
      <canvas id="chart" width="1180" height="560"></canvas>
      <div class="tooltip" id="tooltip"></div>
      <div class="legend">
        <span class="legend-item"><span class="swatch actual"></span>Actual history</span>
        <span class="legend-item"><span class="swatch"></span>Forecast</span>
        <span class="legend-item"><span class="swatch band"></span>P10-P90 interval</span>
      </div>
      <div class="note">History shows the last 365 days before the submission horizon. Intervals are exported in outputs/submission_intervals.csv.</div>
    </section>
  </main>

  <script>
    const DATA = __DATA_JSON__;
    const SUMMARY = __SUMMARY_JSON__;
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const revenueBtn = document.getElementById("revenueBtn");
    const cogsBtn = document.getElementById("cogsBtn");
    let metric = "revenue";
    let hoverIndex = null;

    function money(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return "VND " + Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    function pct(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return (Number(value) * 100).toFixed(1) + "%";
    }
    function r2(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(3);
    }
    function keys() {
      if (metric === "revenue") {
        return { actual: "actualRevenue", pred: "revenue", p10: "revenueP10", p90: "revenueP90", p05: "revenueP05", p95: "revenueP95" };
      }
      return { actual: "actualCOGS", pred: "cogs", p10: "cogsP10", p90: "cogsP90", p05: "cogsP05", p95: "cogsP95" };
    }
    function updateStats() {
      const isRevenue = metric === "revenue";
      document.getElementById("horizon").textContent = `${SUMMARY.forecastStart} to ${SUMMARY.forecastEnd}`;
      document.getElementById("totalValue").textContent = money(isRevenue ? SUMMARY.totalRevenue : SUMMARY.totalCOGS);
      document.getElementById("weeklyMetric").textContent = isRevenue
        ? `${pct(SUMMARY.weeklyRevenueWape)} / ${r2(SUMMARY.weeklyRevenueR2)}`
        : `${pct(SUMMARY.weeklyCogsWape)} / ${r2(SUMMARY.weeklyCogsR2)}`;
      document.getElementById("dailyMetric").textContent = isRevenue
        ? `${pct(SUMMARY.dailyRevenueWape)} / ${r2(SUMMARY.dailyRevenueR2)}`
        : `${pct(SUMMARY.dailyCogsWape)} / ${r2(SUMMARY.dailyCogsR2)}`;
      document.getElementById("daysValue").textContent = SUMMARY.forecastDays.toLocaleString();
      revenueBtn.classList.toggle("active", isRevenue);
      cogsBtn.classList.toggle("active", !isRevenue);
    }
    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }
    function bounds(k) {
      const vals = [];
      DATA.forEach(row => {
        [k.actual, k.pred, k.p10, k.p90, k.p05, k.p95].forEach(key => {
          const v = row[key];
          if (v !== null && Number.isFinite(v)) vals.push(v);
        });
      });
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const pad = (max - min) * 0.08 || max * 0.08 || 1;
      return { min: Math.max(0, min - pad), max: max + pad };
    }
    function drawLine(points, color, width) {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      let open = false;
      points.forEach(p => {
        if (p.y === null) {
          open = false;
          return;
        }
        if (!open) {
          ctx.moveTo(p.x, p.y);
          open = true;
        } else {
          ctx.lineTo(p.x, p.y);
        }
      });
      ctx.stroke();
    }
    function draw() {
      updateStats();
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const pad = { left: 74, right: 24, top: 20, bottom: 42 };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;
      const k = keys();
      const b = bounds(k);
      const x = i => pad.left + (DATA.length <= 1 ? 0 : i * plotW / (DATA.length - 1));
      const y = v => pad.top + (b.max - v) * plotH / (b.max - b.min);

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#667085";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 5; i++) {
        const val = b.min + (b.max - b.min) * i / 5;
        const yy = y(val);
        ctx.beginPath();
        ctx.moveTo(pad.left, yy);
        ctx.lineTo(w - pad.right, yy);
        ctx.stroke();
        ctx.fillText(money(val), pad.left - 8, yy);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      [0, Math.floor(DATA.length * 0.25), Math.floor(DATA.length * 0.5), Math.floor(DATA.length * 0.75), DATA.length - 1].forEach(i => {
        ctx.fillText(DATA[i].date, x(i), h - pad.bottom + 16);
      });

      const bandRows = DATA.map((row, i) => ({
        x: x(i),
        low: row[k.p10] === null ? null : y(row[k.p10]),
        high: row[k.p90] === null ? null : y(row[k.p90])
      })).filter(p => p.low !== null && p.high !== null);
      if (bandRows.length) {
        ctx.fillStyle = "rgba(37, 99, 235, 0.16)";
        ctx.beginPath();
        bandRows.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.high) : ctx.lineTo(p.x, p.high));
        [...bandRows].reverse().forEach(p => ctx.lineTo(p.x, p.low));
        ctx.closePath();
        ctx.fill();
      }

      drawLine(DATA.map((row, i) => ({ x: x(i), y: row[k.actual] === null ? null : y(row[k.actual]) })), "#1f2937", 2);
      drawLine(DATA.map((row, i) => ({ x: x(i), y: row[k.pred] === null ? null : y(row[k.pred]) })), "#2563eb", 2.5);

      const splitIndex = DATA.findIndex(row => row[k.pred] !== null);
      if (splitIndex >= 0) {
        const xx = x(splitIndex);
        ctx.strokeStyle = "#9ca3af";
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(xx, pad.top);
        ctx.lineTo(xx, h - pad.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (hoverIndex !== null) {
        const row = DATA[hoverIndex];
        const xx = x(hoverIndex);
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xx, pad.top);
        ctx.lineTo(xx, h - pad.bottom);
        ctx.stroke();

        const actual = row[k.actual];
        const pred = row[k.pred];
        const anchor = pred !== null ? y(pred) : (actual !== null ? y(actual) : pad.top);
        ctx.fillStyle = "#111827";
        ctx.beginPath();
        ctx.arc(xx, anchor, 3.5, 0, Math.PI * 2);
        ctx.fill();

        tooltip.style.display = "block";
        tooltip.style.left = `${Math.min(Math.max(xx, 110), w - 110)}px`;
        tooltip.style.top = `${Math.max(anchor, 80)}px`;
        tooltip.innerHTML = `
          <strong>${row.date}</strong><br>
          Actual: ${money(actual)}<br>
          Forecast: ${money(pred)}<br>
          P10-P90: ${money(row[k.p10])} - ${money(row[k.p90])}<br>
          P05-P95: ${money(row[k.p05])} - ${money(row[k.p95])}
        `;
      } else {
        tooltip.style.display = "none";
      }
    }
    canvas.addEventListener("mousemove", event => {
      const rect = canvas.getBoundingClientRect();
      const xPos = event.clientX - rect.left;
      const padLeft = 74;
      const padRight = 24;
      const plotW = rect.width - padLeft - padRight;
      const idx = Math.round((xPos - padLeft) * (DATA.length - 1) / plotW);
      hoverIndex = Math.min(Math.max(idx, 0), DATA.length - 1);
      draw();
    });
    canvas.addEventListener("mouseleave", () => {
      hoverIndex = null;
      draw();
    });
    revenueBtn.addEventListener("click", () => {
      metric = "revenue";
      draw();
    });
    cogsBtn.addEventListener("click", () => {
      metric = "cogs";
      draw();
    });
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""
    html = html.replace("__DATA_JSON__", json.dumps(records, separators=(",", ":")))
    html = html.replace("__SUMMARY_JSON__", json.dumps(summary, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"saved {path} ({len(records)} plotted days)")

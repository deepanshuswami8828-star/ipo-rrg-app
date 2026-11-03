"""
IPO RRG app: Relative Rotation Graphs of NSE mainboard IPO stocks (listed from Nov 2024) against NIFTY 50.
Copied from the RRG app (same RRG method). Prices: NSE official files (data_pipeline.py). Daily RRG is the main
view for IPO stocks; the Weekly view appears once a stock has cfg.MIN_WEEKLY_BARS weekly bars.
The Strategy page and the Tags analysis are left out until the strategy is changed (user's decision, 3 Nov 2026).
"""
import os
import sys
import warnings
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

import config as cfg
import data_pipeline as dp
import importlib
import strategies as strat
# The cloud server keeps imported modules in memory between deploys; reload our own modules
# (config first, the others import from it) so a new deploy never runs with stale code.
importlib.reload(cfg)
importlib.reload(dp)
importlib.reload(strat)
from make_rrg_pvrinox import (
    calc_rs_ratio, calc_rs_momentum, quadrant,
    SURFACE, INK_1, INK_2, INK_3, GRID, SERIES, QUAD_BG, QUAD_FG,
    RS_PRESMOOTH, RS_SHORT, RS_LONG, MOM_SHORT, MOM_LONG
)

ORDER = ["Leading", "Weakening", "Lagging", "Improving"]
CYCLES_FROM = pd.Timestamp(cfg.DATA_START_DISPLAY)   # IPO stocks start Nov 2024 or later
SINCE_2024 = "Since listing"

# Color palette for All-Stocks view
STOCK_COLORS = {
    "PVRINOX.NS": "#2a78d6",      # Blue
    "BEL.NS": "#2e7d32",          # Forest Green
    "SUZLON.NS": "#d32f2f",       # Crimson Red
    "WAAREEENER.NS": "#ef6c00",   # Amber Orange
    "BHARTIARTL.NS": "#7b1fa2",   # Purple
    "RELIANCE.NS": "#00838f",     # Deep Cyan
    "PAYTM.NS": "#c2185b",        # Magenta
}
# stocks without a fixed colour above take one from this list, in config order
EXTRA_COLORS = ["#1565c0", "#558b2f", "#ad1457", "#6a1b9a", "#ff8f00", "#00695c", "#4e342e", "#c62828",
                "#283593", "#9e9d24", "#0277bd", "#d84315", "#37474f", "#7cb342", "#5e35b1", "#f9a825",
                "#00897b", "#8d6e63", "#e53935", "#3949ab", "#43a047", "#fb8c00", "#039be5", "#8e24aa"]


def stock_color(ticker: str) -> str:
    if ticker in STOCK_COLORS:
        return STOCK_COLORS[ticker]
    extra = [t for t in cfg.STOCKS if t not in STOCK_COLORS]
    return EXTRA_COLORS[extra.index(ticker) % len(EXTRA_COLORS)] if ticker in extra else SERIES

TF_CONFIG = {
    "Daily": dict(unit="session", units="sessions", warmup=cfg.WARMUP_DAILY, tail=cfg.TAIL_DAYS_DAILY,
                  windows=[60, 125, 250, 500, 750], window=250, slider="Session: ",
                  head="%a %d %b %Y"),
    "Weekly": dict(unit="week", units="weeks", warmup=cfg.WARMUP_WEEKLY, tail=cfg.TAIL_WEEKS,
                   windows=[26, 52, 104, 156, 260], window=52, slider="Week ending: ",
                   head="Week ending %a %d %b %Y"),
}

st.set_page_config(page_title="IPO RRG", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
iframe[data-testid="stIFrame"] {height: calc(100vh - 245px) !important; min-height: 430px;}
div[data-testid="stMetric"] {padding: 0;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.status-pill {padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem;}
.status-live {background: #e8f5e9; color: #2e7d32;}
.status-cached {background: #e3f2fd; color: #1565c0;}
.status-stale {background: #fff3e0; color: #e65100;}
</style>
""", unsafe_allow_html=True)

# Top navigation: two separate tabs. "RRG charts" is the rest of this script; "Tags analysis" is its own page.
nav = st.navigation([
    st.Page(lambda: None, title="RRG charts", icon="📈", url_path="rrg", default=True),
    st.Page("ipo_page.py", title="IPO details", icon="🧾", url_path="ipo"),
], position="top")
nav.run()
if nav.url_path == "ipo":
    st.stop()


# ---- Data Loading & Caching -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_aligned_data(ticker: str, manifest_version: str):
    """
    Loads verified clean stock and benchmark data, performing strict inner join.
    Cached on manifest_version.
    """
    stock_df = dp.load_clean_data(ticker)
    bench_df = dp.load_clean_data(cfg.BENCHMARK_TICKER)
    if stock_df is None or bench_df is None:
        return None
    stock_name = cfg.STOCKS.get(ticker, {}).get("name", ticker)
    aligned = dp.align_stock_with_benchmark(stock_df, bench_df, stock_name, cfg.BENCHMARK_NAME)
    return aligned


def get_stock_data(ticker: str) -> Tuple[Optional[pd.DataFrame], str, datetime]:
    """Retrieves aligned stock data with status source and timestamp."""
    manifest = dp.load_manifest()
    m_ver = manifest.get("version_id", "none")
    aligned = get_cached_aligned_data(ticker, m_ver)
    updated_at_str = manifest.get("updated_at")
    fetched_dt = datetime.fromisoformat(updated_at_str) if updated_at_str else datetime.now(cfg.IST)
    source = "NSE official data (stored, checked)"
    return aligned, source, fetched_dt


def compute_rrg(df_aligned: pd.DataFrame, stock_name: str, tf: str = "Daily") -> Tuple[pd.DataFrame, bool]:
    """Computes RRG daily or weekly series from aligned closes."""
    bench_name = cfg.BENCHMARK_NAME
    closes = df_aligned.copy()
    if tf == "Weekly":
        # Group by Friday week ending; take last trading day of each week
        closes = closes.groupby(closes.index.to_period("W-FRI")).tail(1)

    raw_len = len(closes)
    warmup = min(TF_CONFIG[tf]["warmup"], raw_len // 3)
    if raw_len < warmup + 10:
        return pd.DataFrame(), False

    rr = calc_rs_ratio(closes[stock_name], closes[bench_name])
    rm = calc_rs_momentum(rr)
    df = pd.DataFrame({"RS_Ratio": rr, "RS_Momentum": rm, "Close": closes[stock_name]}).dropna()
    df = df.iloc[warmup:]
    df["Quadrant"] = [quadrant(a, b) for a, b in zip(df["RS_Ratio"], df["RS_Momentum"])]

    # Weekly minimum bar check (section 7.3)
    is_sufficient = len(df) >= cfg.MIN_WEEKLY_BARS if tf == "Weekly" else len(df) >= 30
    return df, is_sufficient


def weekly_asof(closes: pd.DataFrame, stock_name: str, bench_name: str = cfg.BENCHMARK_NAME) -> pd.DataFrame:
    """
    Weekly RS-Ratio / RS-Momentum as they stood on each trading day (no look-ahead).
    """
    week = closes.index.to_period("W-FRI")
    wk = closes.groupby(week).tail(1)
    if len(wk) < 15:
        return pd.DataFrame()

    rs_w = pd.Series((wk[stock_name] / wk[bench_name]).to_numpy())
    sm = rs_w.rolling(RS_PRESMOOTH, min_periods=1).mean()
    e_fast = sm.ewm(span=RS_SHORT, adjust=False).mean().to_numpy()
    e_slow = sm.ewm(span=RS_LONG, adjust=False).mean().to_numpy()
    rr_w = pd.Series(100.0 * e_fast / e_slow)
    m_fast = rr_w.ewm(span=MOM_SHORT, adjust=False).mean().to_numpy()
    m_slow = rr_w.ewm(span=MOM_LONG, adjust=False).mean().to_numpy()
    prior = rs_w.rolling(RS_PRESMOOTH - 1, min_periods=1).sum().to_numpy()

    pos = pd.Index(wk.index.to_period("W-FRI")).get_indexer(week)
    prev = pos - 1
    warmup = min(TF_CONFIG["Weekly"]["warmup"], len(wk) // 3)
    ok = prev >= warmup
    if not ok.any():
        return pd.DataFrame()

    prev_ok = prev[ok]
    d_idx = closes.index[ok]
    rs_d = (closes[stock_name] / closes[bench_name]).to_numpy()[ok]

    def step(x, state, span):
        a = 2.0 / (span + 1.0)
        return a * x + (1 - a) * state

    sm_d = (prior[prev_ok] + rs_d) / RS_PRESMOOTH
    rr_d = 100.0 * step(sm_d, e_fast[prev_ok], RS_SHORT) / step(sm_d, e_slow[prev_ok], RS_LONG)
    rm_d = 100.0 * step(rr_d, m_fast[prev_ok], MOM_SHORT) / step(rr_d, m_slow[prev_ok], MOM_LONG)
    out = pd.DataFrame({"RS_Ratio": rr_d, "RS_Momentum": rm_d,
                        "Close": closes[stock_name].to_numpy()[ok]}, index=d_idx)
    out["Quadrant"] = [quadrant(a, b) for a, b in zip(out["RS_Ratio"], out["RS_Momentum"])]
    return out


# ---- UI Helpers -------------------------------------------------------------
def periods(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive runs of the same quadrant -> one row per run, newest first."""
    run_id = (df["Quadrant"] != df["Quadrant"].shift()).cumsum()
    rows = [{"Quadrant": g["Quadrant"].iloc[0], "From": g.index[0], "To": g.index[-1],
             "Bars": len(g)} for _, g in df.groupby(run_id)]
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def frame_range(xs, ys):
    """Axis limits for trail, keeping the 100/100 crosshair clearly in view."""
    spread = max(xs.max() - xs.min(), ys.max() - ys.min())
    pad, margin = max(0.35, 0.20 * spread), max(0.45, 0.25 * spread)
    return ([min(xs.min() - pad, 100 - margin), max(xs.max() + pad, 100 + margin)],
            [min(ys.min() - pad, 100 - margin), max(ys.max() + pad, 100 + margin)])


def r3(values):
    return [round(float(v), 3) for v in values]


def sparse_labels(xs, ys, xr, yr, gap=0.06, aspect=0.6):
    n = len(xs)
    px = [(x - xr[0]) / max(xr[1] - xr[0], 1e-4) for x in xs]
    py = [(y - yr[0]) / max(yr[1] - yr[0], 1e-4) * aspect for y in ys]
    keep = [n - 1, 0]
    for i in range(n - 2, 0, -1):
        if all((px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2 > gap ** 2 for j in keep):
            keep.append(i)
    return [str(i + 1) if i in keep else "" for i in range(n)]


def quad_corners(xr, yr, s, size):
    corner = {"Improving": (xr[0], yr[1], "left", "top"),
              "Leading": (xr[1], yr[1], "right", "top"),
              "Lagging": (xr[0], yr[0], "left", "bottom"),
              "Weakening": (xr[1], yr[0], "right", "bottom")}
    return [dict(x=x, y=y, xref="x" + s, yref="y" + s, xanchor=xa, yanchor=ya,
                 showarrow=False, xshift=8 if xa == "left" else -8,
                 yshift=-6 if ya == "top" else 6, text=f"<b>{q.upper()}</b>",
                 font=dict(color=QUAD_FG[q], size=size))
            for q, (x, y, xa, ya) in corner.items()]


def panel_parts(view, s, xr, yr, aspect, small):
    tail, dates, last_text = view["tail"], view["dates"], view["last_text"]
    xs, ys = tail["RS_Ratio"].to_numpy(), tail["RS_Momentum"].to_numpy()
    quads = list(tail["Quadrant"])
    n = len(tail)
    fade = [round(0.3 + 0.7 * i / max(n - 1, 1), 2) for i in range(n)]
    hover = [f"<b>#{i + 1}  {d}</b><br>{q}<br>RS-Ratio {r:.2f}<br>RS-Momentum {m:.2f}<br>Close ₹{c:.2f}"
             for i, (d, r, m, q, c) in enumerate(zip(dates, xs, ys, quads, tail["Close"]))]
    x, y = r3(xs), r3(ys)
    ax = dict(type="scatter", xaxis="x" + s, yaxis="y" + s)
    traces = [
        dict(x=x, y=y, mode="lines", line=dict(color=SERIES, width=2),
             opacity=0.55, hoverinfo="skip", **ax),
        dict(x=x, y=y, mode="markers+text",
             text=sparse_labels(xs, ys, xr, yr, aspect=aspect), textposition="top center",
             textfont=dict(color=INK_2, size=10 if small else 11),
             marker=dict(size=10 if small else 11, color=[QUAD_FG[q] for q in quads],
                         opacity=fade, line=dict(color=SURFACE, width=1.2)),
             hovertext=hover, hoverinfo="text", **ax),
        dict(x=x[-1:], y=y[-1:], mode="markers",
             marker=dict(size=19 if small else 22, color=QUAD_FG[quads[-1]],
                         line=dict(color=SURFACE, width=2.5)),
             hovertext=hover[-1:], hoverinfo="text", **ax),
    ]
    ref = dict(xref="x" + s, yref="y" + s, showarrow=False)
    notes = quad_corners(xr, yr, s, 11 if small else 13) + [
        dict(x=x[0], y=y[0], yshift=-18, text=tail.index[0].strftime("%d %b"),
             font=dict(color=INK_3, size=10 if small else 11), **ref),
        dict(x=x[-1], y=y[-1], yshift=-22, text=f"<b>{last_text}</b>",
             font=dict(color=INK_1, size=11 if small else 12), **ref),
    ]
    heading = (f"<b>{view['heading']}</b>{'<br>' if small else '  ·  '}"
               f"<span style='color:{QUAD_FG[quads[-1]]}'><b>{quads[-1].upper()}</b></span>  ·  "
               f"RS-Ratio {xs[-1]:.2f}  ·  RS-Momentum {ys[-1]:.2f}")
    return traces, notes, heading


FIT_AND_KEYS = """
var gd = document.getElementById('{plot_id}');
document.documentElement.style.height = '100%';
document.body.style.height = '100%';
document.body.style.margin = '0';
gd.parentElement.style.height = '100%';
gd.style.height = '100%';
Plotly.Plots.resize(gd);
var names = gd._transitionData._frames.map(function (f) { return f.name; });
var at = names.length - 1;
function go(i) {
    at = Math.max(0, Math.min(names.length - 1, i));
    Plotly.animate(gd, [names[at]], {mode: 'immediate', frame: {duration: 0, redraw: true},
                                     transition: {duration: 0}});
}
gd.on('plotly_sliderchange', function (e) { at = names.indexOf(e.step.label); });
gd.on('plotly_animatingframe', function (e) { at = names.indexOf(e.name); });
document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { go(at - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { go(at + 1); e.preventDefault(); }
});
"""


def rrg_figure(panels, names, slider_prefix):
    two = len(panels) == 2
    suffix = ["", "2"][:len(panels)]
    domains = [[0, 0.47], [0.53, 1]] if two else [[0, 1]]
    aspect = 0.95 if two else 0.6

    def frame_parts(j):
        data, notes, layout = [], [], {}
        for p, s, dom in zip(panels, suffix, domains):
            v = p["views"][j]
            xr, yr = p["fixed"] or frame_range(v["tail"]["RS_Ratio"].to_numpy(),
                                               v["tail"]["RS_Momentum"].to_numpy())
            xr, yr = r3(xr), r3(yr)
            traces, ann, heading = panel_parts(v, s, xr, yr, aspect, two)
            data += traces
            notes += ann
            layout["xaxis" + s] = dict(range=xr)
            layout["yaxis" + s] = dict(range=yr)
            if two:
                notes.append(dict(x=(dom[0] + dom[1]) / 2, y=1.0, xref="paper", yref="paper",
                                  xanchor="center", yanchor="bottom", yshift=6,
                                  showarrow=False, align="center",
                                  text=f"<span style='font-size:15px'><b>{p['name'].upper()}</b></span>  {heading}",
                                  font=dict(size=12, color=INK_2)))
            else:
                layout["title"] = dict(text=heading, y=0.985, yanchor="top", font=dict(size=15))
        layout["annotations"] = notes
        return data, layout

    frames = [dict(data=frame_parts(j)[0], layout=frame_parts(j)[1], name=name)
              for j, name in enumerate(names)]
    data, layout = frame_parts(len(names) - 1)

    big = 1e4
    shapes = []
    for s in suffix:
        ref = dict(xref="x" + s, yref="y" + s, layer="below")
        for q, (x0, x1, y0, y1) in {"Improving": (-big, 100, 100, big),
                                    "Leading": (100, big, 100, big),
                                    "Lagging": (-big, 100, -big, 100),
                                    "Weakening": (100, big, -big, 100)}.items():
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                               fillcolor=QUAD_BG[q], line=dict(width=0), **ref))
        shapes.append(dict(type="line", x0=-big, x1=big, y0=100, y1=100,
                           line=dict(color=INK_3, width=1.1), **ref))
        shapes.append(dict(type="line", x0=100, x1=100, y0=-big, y1=big,
                           line=dict(color=INK_3, width=1.1), **ref))

    step = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
    xt = f"RS-Ratio (100 = in line with {cfg.BENCHMARK_NAME})" if two else f"JdK RS-Ratio — relative strength level (100 = in line with {cfg.BENCHMARK_NAME})"
    yt = "RS-Momentum" if two else "JdK RS-Momentum — rate of change of relative strength"
    for s, dom in zip(suffix, domains):
        layout["xaxis" + s].update(domain=dom, anchor="y" + s, gridcolor=GRID, zeroline=False,
                                   title=dict(text=xt, standoff=4, font=dict(size=12)))
        layout["yaxis" + s].update(anchor="x" + s, gridcolor=GRID, zeroline=False,
                                   title=dict(text=yt, font=dict(size=12)))
    layout.update(
        shapes=shapes, autosize=True, showlegend=False, hovermode="closest",
        margin=dict(l=65, r=20, t=58 if two else 40, b=85),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2),
        sliders=[dict(active=len(frames) - 1, x=0, len=1, y=0, yanchor="top",
                      steps=[dict(method="animate", label=f["name"], args=[[f["name"]], step])
                             for f in frames],
                      pad=dict(t=42, b=0, l=0, r=0),
                      currentvalue=dict(prefix=slider_prefix, xanchor="left",
                                        font=dict(size=13, color=INK_1)),
                      ticklen=0, minorticklen=0, font=dict(size=1, color=SURFACE))],
    )
    return dict(data=data, layout=layout, frames=frames)


def fixed_range(views):
    every = pd.concat([v["tail"] for v in views])
    return frame_range(every["RS_Ratio"].to_numpy(), every["RS_Momentum"].to_numpy())


def single_views(df, tf, window, tail_n, provisional_last):
    head = TF_CONFIG[tf]["head"]
    start = max(len(df) - window, tail_n - 1)
    views = []
    for k in range(start, len(df)):
        tail = df.iloc[k - tail_n + 1:k + 1]
        day = tail.index[-1]
        tag = " (provisional)" if provisional_last and k == len(df) - 1 else ""
        views.append(dict(tail=tail, dates=[d.strftime(head) for d in tail.index],
                          last_text=day.strftime("%d %b %Y") + tag,
                          heading=day.strftime(head)))
    return views, [df.index[k].strftime("%Y-%m-%d") for k in range(start, len(df))]


def weekly_views_on(days, wk, asof, tail_w, open_week):
    wk_per = wk.index.to_period("W-FRI")
    head = TF_CONFIG["Weekly"]["head"]
    views = []
    for d in days:
        i = wk_per.searchsorted(d.to_period("W-FRI"))
        done = wk.iloc[max(0, i - (tail_w - 1)):i]
        now_row = asof.loc[[d]] if d in asof.index else wk.iloc[-1:]
        tail = pd.concat([done[now_row.columns], now_row])
        week_final = i < len(wk) and wk.index[i] == d and not (open_week and i == len(wk) - 1)
        last = ("Week ending %s" if week_final else "Week to %s") % d.strftime("%a %d %b %Y")
        views.append(dict(tail=tail, dates=[x.strftime(head) for x in done.index] + [last],
                          last_text=d.strftime("%d %b %Y") + ("" if week_final else " (week so far)"),
                          heading=last))
    return views


@st.cache_data(max_entries=8, show_spinner=False)
def chart_html(key, _build):
    return pio.to_html(_build(), validate=False, include_plotlyjs=True, full_html=True,
                       auto_play=False, default_width="100%", default_height="100%",
                       config={"displaylogo": False, "responsive": True},
                       post_script=FIT_AND_KEYS)


def embed(key, build):
    st.iframe(chart_html(key, build), height=700)


def quad_style(col):
    return [f"background-color: {QUAD_BG[q]}; color: {QUAD_FG[q]}; font-weight: bold"
            if q in QUAD_BG else "" for q in col]


# ---- Cycles & Timeline ------------------------------------------------------
UP = ("Improving", "Leading")
UP_FG, DOWN_FG = QUAD_FG["Leading"], QUAD_FG["Lagging"]
WEEK_DOWN = "#e8893a"
WEEK_STRIP = {"Leading": "#2e7d4f", "Improving": "#8fd19e",
              "Weakening": WEEK_DOWN, "Lagging": WEEK_DOWN}


def phases(df):
    up = df["Quadrant"].isin(UP)
    run_id = (up != up.shift()).cumsum()
    out = []
    for _, g in df.groupby(run_id):
        route = [q for i, q in enumerate(g["Quadrant"]) if i == 0 or q != g["Quadrant"].iloc[i - 1]]
        out.append(dict(up=bool(g["Quadrant"].iloc[0] in UP), start=g.index[0], last=g.index[-1],
                        bars=len(g), entered=g["Quadrant"].iloc[0], route=route))
    for a, b in zip(out, out[1:]):
        a["next"] = b["start"]
    if out:
        out[-1]["next"] = None
    return out


def moves(df):
    ph = phases(df)
    last_close = df["Close"].iloc[-1]
    out = []
    for prev, p in zip(ph, ph[1:]):
        close = df.loc[p["start"], "Close"]
        end_close = df.loc[p["next"], "Close"] if p["next"] is not None else last_close
        out.append(dict(date=p["start"], up=p["up"], frm=prev["route"][-1], to=p["entered"],
                        route=p["route"], bars=p["bars"], open=p["next"] is None,
                        until=p["next"], close=close,
                        change=100 * (end_close / close - 1)))
    return out


def cycle_strip(items, start):
    fig = go.Figure()
    names = [n for n, _ in items]
    for row, (name, df) in enumerate(items):
        y0, y1 = row + 0.12, row + 0.88
        end = df.index[-1]
        for p in phases(df):
            x0 = max(p["start"], start)
            x1 = p["next"] or end + pd.Timedelta(days=1)
            if x1 <= start:
                continue
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, opacity=0.85,
                          fillcolor=UP_FG if p["up"] else DOWN_FG, line_width=0)
            fig.add_trace(go.Scatter(
                x=[x0 + (x1 - x0) / 2], y=[row + 0.5], mode="markers",
                marker=dict(size=18, opacity=0), showlegend=False,
                hovertemplate=f"<b>{name} · {'UP' if p['up'] else 'DOWN'}</b><br>"
                              f"{p['start'].strftime('%d %b %Y')} → {p['next'].strftime('%d %b %Y') if p['next'] else 'now'}<br>"
                              f"{p['bars']} {'weeks' if name == 'Weekly' else 'sessions'}<br>"
                              f"{' → '.join(p['route'])}<extra></extra>"))
    for label, color in (("Up: Improving / Leading", UP_FG), ("Down: Weakening / Lagging", DOWN_FG)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(size=12, symbol="square", color=color)))
    fig.update_layout(height=90 + 55 * len(items), margin=dict(l=65, r=20, t=30, b=30),
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      hovermode="closest")
    fig.update_xaxes(range=[start, max(df.index[-1] for _, df in items) + pd.Timedelta(days=3)],
                     gridcolor=GRID, dtick="M3", tickformat="%b %Y")
    fig.update_yaxes(range=[len(items), 0], tickvals=[i + 0.5 for i in range(len(items))],
                     ticktext=[f"<b>{n}</b>" for n in names], showgrid=False, zeroline=False)
    return fig


def timeline_figure(seg, tf):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
    fig.add_hline(y=100, line_color=INK_3, line_width=1, row=1, col=1)
    fig.add_trace(go.Scatter(x=seg.index, y=seg["RS_Ratio"], name="RS-Ratio",
                             line=dict(color=SERIES, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg.index, y=seg["RS_Momentum"], name="RS-Momentum",
                             line=dict(color="#d0782a", width=2)), row=1, col=1)
    head = TF_CONFIG[tf]["head"]
    for q in ORDER:
        part = seg[seg["Quadrant"] == q]
        fig.add_trace(go.Bar(x=part.index, y=[1] * len(part), name=q, marker_color=QUAD_FG[q],
                             marker_line_width=0,
                             customdata=part[["RS_Ratio", "RS_Momentum"]].to_numpy(),
                             hovertemplate=f"%{{x|{head}}}<br><b>{q}</b><br>RS-Ratio %{{customdata[0]:.2f}}<br>RS-Momentum %{{customdata[1]:.2f}}<extra></extra>"),
                      row=2, col=1)
    fig.update_layout(height=360, bargap=0, barmode="stack", hovermode="x unified",
                      margin=dict(l=65, r=20, t=10, b=10),
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2),
                      legend=dict(orientation="h", y=1.08, x=0, traceorder="normal"))
    if tf == "Daily":
        holidays = pd.bdate_range(seg.index[0], seg.index[-1]).difference(seg.index)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=list(holidays))])
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    return fig


def move_log(df, unit_s, unit_p):
    log = [m for m in moves(df) if m["date"] >= CYCLES_FROM]
    plural = lambda n: f"{n} {unit_s if n == 1 else unit_p}"
    cur = phases(df)[-1]
    ups = [m["bars"] for m in log if m["up"] and not m["open"]]
    downs = [m["bars"] for m in log if not m["up"] and not m["open"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Now", f"{'Up' if cur['up'] else 'Down'} · {cur['route'][-1]}",
              f"since {cur['start'].strftime('%d %b %Y')}",
              delta_color="normal" if cur["up"] else "inverse")
    c2.metric("Moves up since listing", sum(m["up"] for m in log))
    c3.metric("Average stay up", f"{(sum(ups) / len(ups)):.1f} {unit_p}" if ups else "—")
    c4.metric("Average stay down", f"{(sum(downs) / len(downs)):.1f} {unit_p}" if downs else "—")

    if not log:
        st.info("No moves since listing.")
        return
    table = pd.DataFrame([{
        "#": i + 1,
        "Date": m["date"].strftime("%a %d %b %Y"),
        "Move": "▲ Up" if m["up"] else "▼ Down",
        "From": m["frm"],
        "Into": m["to"],
        "Stayed": plural(m["bars"]) + (" (still there)" if m["open"] else ""),
        "Until": m["until"].strftime("%d %b %Y") if m["until"] is not None else "now",
        "Close": f"₹{m['close']:.2f}",
        "Price till next move": (f"{m['change']:+.1f}%") if abs(m["change"]) >= 0.05 else "0.0%",
    } for i, m in enumerate(log)])

    def move_style(col):
        return ["color: %s; font-weight: bold" % (UP_FG if v.startswith("▲") else DOWN_FG) for v in col]

    def price_style(col):
        return ["color: %s; font-weight: bold" % (UP_FG if v.startswith("+") else
                                                  DOWN_FG if v.startswith("-") else INK_2) for v in col]

    st.dataframe(table.style.apply(quad_style, subset=["From", "Into"])
                 .apply(move_style, subset=["Move"])
                 .apply(price_style, subset=["Price till next move"]),
                 hide_index=True, width="stretch", height=38 + 35 * len(table))


def cycle_section(items):
    st.subheader("Cycles since listing")
    st.caption("▲ **Up** = entered Improving or Leading (RS-Momentum > 100).  "
               "▼ **Down** = entered Weakening or Lagging (RS-Momentum < 100).")
    st.plotly_chart(cycle_strip(items, CYCLES_FROM), width="stretch", theme=None,
                    config={"displaylogo": False, "displayModeBar": False})
    for name, df in items:
        weekly = name == "Weekly"
        st.markdown(f"##### {name} moves")
        move_log(df, "week" if weekly else "session", "weeks" if weekly else "sessions")


# ---- Header & Status Panel --------------------------------------------------
def render_status_panel(selected_ticker: str):
    """Renders comprehensive Data Status panel for selected stock and all stocks."""
    manifest = dp.load_manifest()
    now_ist = datetime.now(cfg.IST)
    ticker_info = manifest.get("tickers", {}).get(selected_ticker, {})
    last_attempt = manifest.get("last_attempt", {})

    with st.expander("📊 Data Status & Validation Panel", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Selected Stock:** {cfg.STOCKS.get(selected_ticker, {}).get('name', selected_ticker)}")
        c1.caption(f"Ticker: `{selected_ticker}` | Benchmark: `{cfg.BENCHMARK_NAME}`")

        last_success = manifest.get("updated_at")
        success_disp = datetime.fromisoformat(last_success).strftime("%d %b %Y %H:%M IST") if last_success else "None"
        c2.markdown(f"**Last Valid Sync:** {success_disp}")
        c2.caption(f"Provider: {manifest.get('provider', cfg.PROVIDER)}")

        last_res = last_attempt.get("result", "UNKNOWN")
        res_color = "green" if last_res == "SUCCESS" else ("orange" if last_res == "NO_NEW_SESSION" else "red")
        c3.markdown(f"**Last Attempt Result:** <span style='color:{res_color};font-weight:700'>{last_res}</span>", unsafe_allow_html=True)
        c3.caption(f"{last_attempt.get('message', '')[:60]}...")

        expected_sess = dp.get_expected_latest_session(now_ist)
        c4.markdown(f"**Expected Latest Session:** {expected_sess.strftime('%d %b %Y')}")
        c4.caption("Next unattended trigger: Weekdays 15:45 IST")

        # Flagged moves table
        flags = ticker_info.get("flagged_moves", [])
        if flags:
            st.markdown(f"**Flagged One-Day Moves (> ±{cfg.SUSPICIOUS_MOVE_PCT}%) for {cfg.STOCKS.get(selected_ticker, {}).get('name')}:**")
            st.dataframe(pd.DataFrame(flags), hide_index=True)

        # Compact all stocks status table
        st.markdown("**All Configured Stocks Coverage & Alignment:**")
        summary_rows = []
        for t, s_cfg in cfg.STOCKS.items():
            t_meta = manifest.get("tickers", {}).get(t, {})
            summary_rows.append({
                "Stock": s_cfg["name"],
                "Ticker": t,
                "Sector": s_cfg["sector"],
                "Rows": t_meta.get("rows", 0),
                "First Date": t_meta.get("first_date", "—"),
                "Latest Date": t_meta.get("last_date", "—"),
                "Status": t_meta.get("status", "NO_DATA"),
                "Flagged Moves": len(t_meta.get("flagged_moves", []))
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")


def header(title: str, stock_name: str, source: str, fetched: datetime, now: datetime, status_text: str):
    st.markdown(f"#### {stock_name} against {cfg.BENCHMARK_NAME} — {title}")
    market_str = "🟢 NSE open" if dp.is_market_open(now) else "⚪ NSE closed"
    st.caption(f"{market_str} — {status_text}  ·  sync {fetched.strftime('%d %b %Y %H:%M')} IST  ·  {source}")


def formula_note(extra: str):
    st.caption(f"RS-Ratio = 100 × EMA(SMA(RS,{RS_PRESMOOTH}),{RS_SHORT}) / EMA(SMA(RS,{RS_PRESMOOTH}),{RS_LONG})   ·   "
               f"RS-Momentum = 100 × EMA(RS-Ratio,{MOM_SHORT}) / EMA(RS-Ratio,{MOM_LONG})   ·   {extra}   ·   "
               f"computed over full verified history after warm-up")


# ---- All Stocks Page --------------------------------------------------------
def all_stocks_page(timeframe: str, tickers: list, tag: str):
    st.subheader(f"All Stocks Comparison ({timeframe} RRG) · Tag: {tag}")
    st.caption(f"Rotating relative strength of the {len(tickers)} IPO stocks in tag {tag} plotted simultaneously against NIFTY 50. "
               "The chart covers the dates all of them share, so a tag with a recent listing shows a short window.")

    manifest = dp.load_manifest()
    m_ver = manifest.get("version_id", "none")
    stock_dfs = {}
    table_rows = []
    earliest_dates = []

    for ticker in tickers:
        s_info = cfg.STOCKS[ticker]
        aligned = get_cached_aligned_data(ticker, m_ver)
        if aligned is None:
            table_rows.append({
                "Stock": s_info["name"], "Sector": s_info["sector"],
                "Daily Quad": "No data", "Weekly Quad": "No data",
                "RS-Ratio": 0.0, "RS-Mom": 0.0, "Days in Quad": 0, "Latest Date": "—"
            })
            continue

        s_name = s_info["name"]
        d_df, d_ok = compute_rrg(aligned, s_name, tf="Daily")
        w_df, w_ok = compute_rrg(aligned, s_name, tf="Weekly")
        stock_dfs[ticker] = (d_df if timeframe == "Daily" else w_df, s_name)

        if len(d_df) > 0:
            d_last = d_df.iloc[-1]
            d_streak = periods(d_df).iloc[0]["Bars"]
            w_quad = w_df.iloc[-1]["Quadrant"] if (w_ok and len(w_df) > 0) else "Needs history"
            curr_df = d_df if timeframe == "Daily" else w_df
            ratio_val = curr_df.iloc[-1]["RS_Ratio"] if len(curr_df) > 0 else 0.0
            mom_val = curr_df.iloc[-1]["RS_Momentum"] if len(curr_df) > 0 else 0.0
            table_rows.append({
                "Stock": s_name,
                "Sector": s_info["sector"],
                "Daily Quad": d_last["Quadrant"],
                "Weekly Quad": w_quad,
                "RS-Ratio": round(float(ratio_val), 2),
                "RS-Mom": round(float(mom_val), 2),
                "Days in Quad": int(d_streak),
                "Latest Date": d_df.index[-1].strftime("%Y-%m-%d")
            })

    # Summary table
    st.dataframe(pd.DataFrame(table_rows).style.apply(quad_style, subset=["Daily Quad", "Weekly Quad"]),
                 hide_index=True, width="stretch")

    # Multi-stock chart
    valid_stocks = {t: (df, name) for t, (df, name) in stock_dfs.items() if len(df) >= 20}
    if not valid_stocks:
        st.warning("Insufficient history to draw multi-stock RRG chart.")
        return

    # Find shared recent date window
    shared_end = min(df.index[-1] for df, _ in valid_stocks.values())
    shared_start = max(min(df.index[0] for df, _ in valid_stocks.values()), CYCLES_FROM)
    shared_dates = [d for d in next(iter(valid_stocks.values()))[0].index if shared_start <= d <= shared_end]

    # Slider covers the most recent 60 sessions for multi-stock comparison
    shared_dates = shared_dates[-60:]
    tail_n = 15
    frames = []
    step_cmd = dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))

    for target_d in shared_dates:
        traces = []
        for ticker, (df, s_name) in valid_stocks.items():
            color = stock_color(ticker)
            hist = df.loc[:target_d]
            if len(hist) < 2:
                continue
            tail = hist.tail(tail_n)
            xs = r3(tail["RS_Ratio"].to_numpy())
            ys = r3(tail["RS_Momentum"].to_numpy())
            traces.append(dict(
                type="scatter", x=xs, y=ys, mode="lines",
                line=dict(color=color, width=2), hoverinfo="skip", showlegend=False
            ))
            traces.append(dict(
                type="scatter", x=xs[-1:], y=ys[-1:], mode="markers+text", name=s_name,
                text=[s_name], textposition="top center",
                textfont=dict(color=INK_1, size=11),
                marker=dict(size=14, color=color, line=dict(color=SURFACE, width=1.5)),
                hovertemplate=f"<b>{s_name}</b><br>{target_d.strftime('%d %b %Y')}<br>"
                              f"RS-Ratio %{{x:.2f}}<br>RS-Momentum %{{y:.2f}}<extra></extra>"
            ))
        frames.append(dict(data=traces, name=target_d.strftime("%Y-%m-%d")))

    # Initial frame
    data_last = frames[-1]["data"]
    big = 1e4
    shapes = []
    for q, (x0, x1, y0, y1) in {"Improving": (-big, 100, 100, big),
                                "Leading": (100, big, 100, big),
                                "Lagging": (-big, 100, -big, 100),
                                "Weakening": (100, big, -big, 100)}.items():
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                           fillcolor=QUAD_BG[q], line=dict(width=0), layer="below"))
    shapes.append(dict(type="line", x0=-big, x1=big, y0=100, y1=100, line=dict(color=INK_3, width=1.1)))
    shapes.append(dict(type="line", x0=100, x1=100, y0=-big, y1=big, line=dict(color=INK_3, width=1.1)))

    layout = dict(
        shapes=shapes, height=650, margin=dict(l=65, r=20, t=30, b=80),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2),
        xaxis=dict(range=[85, 125], gridcolor=GRID, title=f"JdK RS-Ratio (100 = in line with {cfg.BENCHMARK_NAME})"),
        yaxis=dict(range=[85, 120], gridcolor=GRID, title="JdK RS-Momentum"),
        sliders=[dict(active=len(frames) - 1, x=0, len=1, y=0, yanchor="top",
                      steps=[dict(method="animate", label=f["name"], args=[[f["name"]], step_cmd]) for f in frames],
                      currentvalue=dict(prefix="Session: ", xanchor="left", font=dict(size=13, color=INK_1)))]
    )
    fig_dict = dict(data=data_last, layout=layout, frames=frames)
    st.plotly_chart(fig_dict, width="stretch", config={"displaylogo": False})


# ---- Single Page (Daily / Weekly) -------------------------------------------
def single_page(stock_ticker: str, tf: str):
    stock_info = cfg.STOCKS.get(stock_ticker, {"name": stock_ticker, "sector": ""})
    stock_name = stock_info["name"]

    aligned, source, fetched = get_stock_data(stock_ticker)
    if aligned is None or len(aligned) == 0:
        st.error(f"No clean price data available for {stock_name}. Please press 'Refresh All Data'.")
        return

    df, sufficient = compute_rrg(aligned, stock_name, tf=tf)
    now = datetime.now(cfg.IST)
    is_prov = False

    if tf == "Weekly" and not sufficient:
        first_date = aligned.index[0].strftime("%d %b %Y")
        st.warning(f"Weekly RRG needs more history - {stock_name} listed on {first_date} (minimum {cfg.MIN_WEEKLY_BARS} weekly bars required). "
                   "For IPO stocks the Daily RRG is the main view: choose Daily.")
        return

    unit, units = TF_CONFIG[tf]["unit"], TF_CONFIG[tf]["units"]
    last, prev = df.iloc[-1], df.iloc[-2]
    streak = periods(df).iloc[0]["Bars"]

    header(f"{tf.lower()} RRG", stock_name, source, fetched, now,
           f"latest {unit} is {'provisional' if is_prov else 'final'}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Quadrant on {df.index[-1].strftime('%d %b')}", last["Quadrant"],
              f"{streak} {unit if streak == 1 else units} in a row", delta_color="off")
    c2.metric("RS-Ratio", f"{last['RS_Ratio']:.2f}", f"{last['RS_Ratio'] - prev['RS_Ratio']:+.2f}")
    c3.metric("RS-Momentum", f"{last['RS_Momentum']:.2f}", f"{last['RS_Momentum'] - prev['RS_Momentum']:+.2f}")
    c4.metric(f"{stock_name} close", f"₹{last['Close']:.2f}",
              f"{100 * (last['Close'] / prev['Close'] - 1):+.2f}% vs previous {unit}")

    win = int((df.index >= CYCLES_FROM).sum()) if window == SINCE_2024 else int(window)
    win = min(win, len(df))

    def build():
        views, names = single_views(df, tf, win, tail_n, is_prov)
        return rrg_figure([dict(name=tf, views=views,
                                fixed=fixed_range(views) if fixed_axes else None)],
                          names, TF_CONFIG[tf]["slider"])

    embed((stock_ticker, tf, win, tail_n, fixed_axes, is_prov, str(fetched)), build)

    seg = df.iloc[-win:]
    cycle_section([(tf, df)])
    runs = periods(seg)

    st.subheader(f"Which quadrant on which {'date' if tf == 'Daily' else 'week'}")
    st.plotly_chart(timeline_figure(seg, tf), width="stretch", theme=None, config={"displaylogo": False})

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Look up a date**")
        pick = st.date_input("Date", value=df.index[-1].date(),
                             min_value=seg.index[0].date(), max_value=now.date(),
                             key=f"pick_{stock_ticker}_{tf}")
        ts = pd.Timestamp(pick)
        hit = df.loc[:ts]
        row_date = hit.index[-1] if len(hit) else df.index[-1]
        row = df.loc[row_date]
        st.markdown(f"**{row_date.strftime(TF_CONFIG[tf]['head'])}** — "
                    f"<span style='color:{QUAD_FG[row['Quadrant']]};font-weight:700'>{row['Quadrant'].upper()}</span>  ·  "
                    f"RS-Ratio {row['RS_Ratio']:.2f}  ·  RS-Momentum {row['RS_Momentum']:.2f}",
                    unsafe_allow_html=True)
        counts = seg["Quadrant"].value_counts().reindex(ORDER, fill_value=0)
        st.markdown(f"**{units.capitalize()} per quadrant (last {len(seg)})**")
        st.dataframe(pd.DataFrame({"Quadrant": counts.index, units.capitalize(): counts.values,
                                   "Share": [f"{100 * v / len(seg):.0f}%" for v in counts.values]})
                     .style.apply(quad_style, subset=["Quadrant"]), hide_index=True, width="stretch")
    with right:
        st.markdown("**Quadrant periods (newest first)**")
        show = runs.assign(From=runs["From"].dt.strftime("%d %b %Y"),
                           To=runs["To"].dt.strftime("%d %b %Y")) \
                   .rename(columns={"Bars": units.capitalize()})
        st.dataframe(show.style.apply(quad_style, subset=["Quadrant"]), hide_index=True, width="stretch", height=400)

    with st.expander(f"Every {unit}, one by one"):
        rows = seg.iloc[::-1]
        table = pd.DataFrame({"Date": rows.index.strftime(TF_CONFIG[tf]["head"]),
                              "Close": rows["Close"].round(2).to_numpy(),
                              "RS-Ratio": rows["RS_Ratio"].round(2).to_numpy(),
                              "RS-Mom.": rows["RS_Momentum"].round(2).to_numpy(),
                              "Quadrant": rows["Quadrant"].to_numpy()})
        st.dataframe(table.style.apply(quad_style, subset=["Quadrant"])
                     .format({"Close": "%.2f", "RS-Ratio": "%.2f", "RS-Mom.": "%.2f"}),
                     hide_index=True, width="stretch", height=500)
        clean_name = dp.sanitize_ticker(stock_ticker).lower()
        st.download_button("Download as CSV", seg.to_csv(float_format="%.4f"),
                           file_name=f"rrg_{clean_name}_{tf.lower()}.csv", mime="text/csv")

    formula_note(f"periods are {units}")


# ---- Both Page --------------------------------------------------------------
def both_page(stock_ticker: str):
    stock_info = cfg.STOCKS.get(stock_ticker, {"name": stock_ticker, "sector": ""})
    stock_name = stock_info["name"]

    aligned, source, fetched = get_stock_data(stock_ticker)
    if aligned is None or len(aligned) == 0:
        st.error(f"No clean price data available for {stock_name}.")
        return

    daily, d_ok = compute_rrg(aligned, stock_name, tf="Daily")
    weekly, w_ok = compute_rrg(aligned, stock_name, tf="Weekly")

    if not w_ok or len(weekly) < cfg.MIN_WEEKLY_BARS:
        first_date = aligned.index[0].strftime("%d %b %Y")
        st.warning(f"Weekly RRG needs more history - {stock_name} listed on {first_date} (minimum {cfg.MIN_WEEKLY_BARS} weekly bars required). "
                   "For IPO stocks the Daily RRG is the main view: choose Daily.")
        return

    asof = weekly_asof(aligned, stock_name)
    now = datetime.now(cfg.IST)
    prov_day, prov_week = False, False

    header("daily + weekly RRG", stock_name, source, fetched, now, "latest session final")

    d_last, w_last = daily.iloc[-1], asof.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Daily quadrant on {daily.index[-1].strftime('%d %b')}", d_last["Quadrant"],
              f"RS-Ratio {d_last['RS_Ratio']:.2f} · RS-Mom. {d_last['RS_Momentum']:.2f}", delta_color="off")
    c2.metric(f"Weekly quadrant on {asof.index[-1].strftime('%d %b')}", w_last["Quadrant"],
              f"RS-Ratio {w_last['RS_Ratio']:.2f} · RS-Mom. {w_last['RS_Momentum']:.2f}", delta_color="off")
    prev = daily.iloc[-2]
    c3.metric(f"{stock_name} close", f"₹{d_last['Close']:.2f}",
              f"{100 * (d_last['Close'] / prev['Close'] - 1):+.2f}% vs previous session")

    win_target = int((daily.index >= CYCLES_FROM).sum()) if window == SINCE_2024 else int(window)
    win = min(win_target, int((daily.index >= asof.index[0]).sum()), len(daily))

    def build():
        d_views, names = single_views(daily, "Daily", win, tail_n, prov_day)
        days = [pd.Timestamp(n) for n in names]
        w_views = weekly_views_on(days, weekly, asof, tail_w, prov_week)
        return rrg_figure([dict(name="Daily", views=d_views,
                                fixed=fixed_range(d_views) if fixed_axes else None),
                           dict(name="Weekly", views=w_views,
                                fixed=fixed_range(w_views) if fixed_axes else None)],
                          names, "Session: ")

    embed((stock_ticker, "Both", win, tail_n, tail_w, fixed_axes, str(fetched)), build)

    seg_d = daily.iloc[-win:]
    cycle_section([("Daily", daily), ("Weekly", weekly)])
    both = pd.DataFrame({"Close": seg_d["Close"],
                         "Daily": seg_d["Quadrant"],
                         "D RS-Ratio": seg_d["RS_Ratio"], "D RS-Mom.": seg_d["RS_Momentum"],
                         "Weekly": asof["Quadrant"].reindex(seg_d.index),
                         "W RS-Ratio": asof["RS_Ratio"].reindex(seg_d.index),
                         "W RS-Mom.": asof["RS_Momentum"].reindex(seg_d.index)})

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Look up a date**")
        pick = st.date_input("Date", value=daily.index[-1].date(),
                             min_value=seg_d.index[0].date(), max_value=now.date(),
                             key=f"pick_both_{stock_ticker}")
        hit = both.loc[:pd.Timestamp(pick)]
        day, row = hit.index[-1], hit.iloc[-1]
        st.markdown(f"**{day.strftime('%a %d %b %Y')}**")
        for name, q, r, m in (("Daily", row["Daily"], row["D RS-Ratio"], row["D RS-Mom."]),
                              ("Weekly", row["Weekly"], row["W RS-Ratio"], row["W RS-Mom."])):
            st.markdown(f"{name} — <span style='color:{QUAD_FG[q]};font-weight:700'>{q.upper()}</span>  ·  "
                        f"RS-Ratio {r:.2f}  ·  RS-Momentum {m:.2f}", unsafe_allow_html=True)
    with right:
        agree = (both["Daily"] == both["Weekly"]).mean()
        st.markdown(f"**Sessions per quadrant (last {len(both)} sessions)**")
        st.dataframe(pd.DataFrame({
            "Quadrant": ORDER,
            "Daily": both["Daily"].value_counts().reindex(ORDER, fill_value=0).to_numpy(),
            "Weekly": both["Weekly"].value_counts().reindex(ORDER, fill_value=0).to_numpy()})
            .style.apply(quad_style, subset=["Quadrant"]), hide_index=True, width="stretch")
        st.caption(f"Daily and weekly agree on {100 * agree:.0f}% of these sessions.")

    with st.expander("Every session: daily and weekly quadrant side by side", expanded=False):
        rows = both.iloc[::-1].reset_index()
        rows["Date"] = rows["Date"].dt.strftime("%a %d %b %Y")
        st.dataframe(rows.style.apply(quad_style, subset=["Daily", "Weekly"]),
                     hide_index=True, width="stretch", height=500)
        clean_name = dp.sanitize_ticker(stock_ticker).lower()
        st.download_button("Download as CSV", both.to_csv(float_format="%.4f"),
                           file_name=f"rrg_{clean_name}_daily_weekly.csv", mime="text/csv")


# ---- Strategy Page ----------------------------------------------------------
def inr(v, sign=False):
    neg = v < 0
    n = int(round(abs(v)))
    s = str(n)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    prefix = "-" if neg else ("+" if sign and n > 0 else "")
    return f"{prefix}₹{s}"


def pct(v):
    return f"{v:+.1f}%" if abs(v) >= 0.05 else "0.0%"


def simulate(daily, trades, start, capital, cost_pct):
    c = cost_pct / 100.0
    close = daily["Close"][daily.index >= start]
    cash, book = float(capital), []
    shares_by_day = pd.Series(0, index=close.index)
    for t in trades:
        px_in = t["buy_px"]
        shares = int(cash // (px_in * (1 + c)))
        paid = shares * px_in * (1 + c)
        px_out = t["sell_px"]
        got = shares * px_out * (1 - c)
        charges = shares * px_in * c + shares * px_out * c
        cash_before = cash
        cash = cash - paid + got
        end = t["sell"] if t["sell"] is not None else close.index[-1]
        shares_by_day[(close.index >= t["buy"]) & (close.index < end)] = shares
        if t["sell"] is None:
            shares_by_day[close.index == end] = shares
        book.append(dict(t, shares=shares, paid=paid, got=got, pnl=got - paid,
                         pnl_pct=100 * (got - paid) / paid if paid else 0.0,
                         charges=charges, cash_before=cash_before, cash_after=cash))

    cash_by_day = pd.Series(float(capital), index=close.index)
    for b in book:
        end = b["sell"] if b["sell"] is not None else close.index[-1]
        inside = (close.index >= b["buy"]) & (close.index < end)
        cash_by_day[inside] = b["cash_before"] - b["paid"]
        cash_by_day[close.index >= end] = b["cash_after"]
        if b["sell"] is None:
            cash_by_day[close.index == end] = b["cash_before"] - b["paid"]
    value = cash_by_day + shares_by_day * close * (1 - c)
    if book and book[-1]["sell"] is None:
        value.iloc[-1] = book[-1]["cash_after"]

    bh_shares = int(capital // (close.iloc[0] * (1 + c)))
    bh_left = capital - bh_shares * close.iloc[0] * (1 + c)
    bh_value = bh_left + bh_shares * close * (1 - c)
    return book, cash, value, bh_value, bh_shares


def strategy_figure(daily, asof, book, value, bh_value, start, capital, stock_name):
    seg = daily[daily.index >= start]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.06, 0.44],
                        vertical_spacing=0.05,
                        subplot_titles=(f"{stock_name} close — green bands = money in the stock",
                                        "Weekly RRG on each day — dark green = Leading, light green = Improving, orange = Weakening / Lagging",
                                        f"Your {inr(capital)} over time — strategy vs buy & hold"))
    wk = asof[(asof.index >= seg.index[0]) & (asof.index <= seg.index[-1])]
    run_id = (wk["Quadrant"] != wk["Quadrant"].shift()).cumsum()
    runs = [(g.index[0], g["Quadrant"].iloc[0]) for _, g in wk.groupby(run_id)]
    for i, (x0, q) in enumerate(runs):
        x1 = runs[i + 1][0] if i + 1 < len(runs) else wk.index[-1] + pd.Timedelta(days=1)
        fig.add_shape(type="rect", xref="x2", yref="y2 domain", x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=WEEK_STRIP[q], opacity=0.9, line_width=0)
    fig.add_trace(go.Scatter(
        x=wk.index, y=[0.5] * len(wk), mode="markers", marker=dict(size=8, opacity=0),
        showlegend=False, customdata=wk[["Quadrant", "RS_Ratio", "RS_Momentum"]].to_numpy(),
        hovertemplate="%{x|%a %d %b %Y}<br>Weekly: <b>%{customdata[0]}</b><br>"
                      "RS-Ratio %{customdata[1]:.2f} · RS-Mom. %{customdata[2]:.2f}<extra></extra>"),
        row=2, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=2, col=1)
    for b in book:
        end = b["sell"] if b["sell"] is not None else seg.index[-1]
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=b["buy"], x1=end, y0=0, y1=1,
                      fillcolor=QUAD_FG["Leading"], opacity=0.14, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=seg.index, y=seg["Close"], name="Close",
                             line=dict(color=INK_2, width=1.4),
                             hovertemplate="%{x|%a %d %b %Y}<br>₹%{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[b["buy"] for b in book], y=[b["buy_px"] for b in book], mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=13, color=QUAD_FG["Leading"], line=dict(color=SURFACE, width=1)),
        hovertext=[f"<b>BUY #{i + 1}</b>  {b['buy'].strftime('%a %d %b %Y')}<br>{b['shares']} shares at ₹{b['buy_px']:.2f}<br>paid {inr(b['paid'])}"
                   for i, b in enumerate(book)], hoverinfo="text"), row=1, col=1)
    sold = [(i, b) for i, b in enumerate(book) if b["sell"] is not None]
    fig.add_trace(go.Scatter(
        x=[b["sell"] for _, b in sold], y=[b["sell_px"] for _, b in sold], mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=13, color=QUAD_FG["Lagging"], line=dict(color=SURFACE, width=1)),
        hovertext=[f"<b>SELL #{i + 1}</b>  {b['sell'].strftime('%a %d %b %Y')}<br>{b['shares']} shares at ₹{b['sell_px']:.2f}<br>got {inr(b['got'])} · {inr(b['pnl'], sign=True)}"
                   for i, b in sold], hoverinfo="text"), row=1, col=1)
    fig.add_trace(go.Scatter(x=value.index, y=value, name="Strategy",
                             line=dict(color=SERIES, width=2.2),
                             hovertemplate="%{x|%d %b %Y}<br>Strategy ₹%{y:,.0f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=bh_value.index, y=bh_value, name="Buy & hold",
                             line=dict(color=INK_3, width=1.5, dash="dot"),
                             hovertemplate="%{x|%d %b %Y}<br>Buy & hold ₹%{y:,.0f}<extra></extra>"), row=3, col=1)
    fig.add_hline(y=capital, line_color=INK_3, line_width=1, line_dash="dash", row=3, col=1,
                  annotation_text="money put in", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=INK_3))
    fig.update_layout(height=760, margin=dict(l=75, r=20, t=75, b=10), hovermode="closest",
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK_2),
                      legend=dict(orientation="h", y=0.995, yanchor="top", yref="container", x=0))
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=13, color=INK_1), x=0, xanchor="left")
    fig.update_xaxes(gridcolor=GRID, dtick="M3", tickformat="%b %Y")
    fig.update_yaxes(gridcolor=GRID, tickprefix="₹", row=1, col=1)
    fig.update_yaxes(gridcolor=GRID, tickprefix="₹", tickformat=",.0f", row=3, col=1)
    return fig


def status_box(color, title, lines, big=None):
    st.markdown(
        f"<div style='border-left:6px solid {color};background:{SURFACE if color == INK_3 else color + '14'};padding:14px 18px;border-radius:6px'>"
        f"<div style='font-size:1.45rem;font-weight:700;color:{color}'>{title}</div>"
        f"{f'<div style=\"font-size:1.15rem;margin-top:4px;color:{INK_1}\">{big}</div>' if big else ''}"
        f"<div style='color:{INK_1};margin-top:4px;line-height:1.6'>{'<br>'.join(lines)}</div></div>",
        unsafe_allow_html=True)


def strategy_page(stock_ticker: str, strategy_id: int = 1):
    stock_info = cfg.STOCKS.get(stock_ticker, {"name": stock_ticker, "sector": ""})
    stock_name = stock_info["name"]

    aligned, source, fetched = get_stock_data(stock_ticker)
    if aligned is None or len(aligned) == 0:
        st.error(f"No clean price data available for {stock_name}.")
        return

    daily, d_ok = compute_rrg(aligned, stock_name, tf="Daily")
    weekly, w_ok = compute_rrg(aligned, stock_name, tf="Weekly")

    if not w_ok or len(weekly) < 60:
        first_date = aligned.index[0].strftime("%d %b %Y")
        st.warning(f"Weekly RRG needs more history - {stock_name} listed on {first_date} (minimum 60 weekly bars required for weekly confirmation).")
        return

    asof = weekly_asof(aligned, stock_name)
    now = datetime.now(cfg.IST)
    start = max(pd.Timestamp(bt_from), daily.index[1], asof.index[0])

    header("strategy", stock_name, source, fetched, now, "signals are final")

    selected_s_id = strategy_id
    meta = strat.STRATEGY_REGISTRY[selected_s_id]

    # Prominent Strategy Explanation Card
    st.markdown(
        f"""
        <div style='background:{SURFACE};border:1px solid {GRID};border-left:5px solid {QUAD_FG["Leading"]};padding:14px 18px;border-radius:8px;margin-top:6px;margin-bottom:14px'>
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
                <span style='font-size:1.25rem;font-weight:700;color:{INK_1}'>{meta['label']}: {meta['name']}</span>
                <span style='background:#e8f5e9;color:#2e7d32;padding:3px 10px;border-radius:12px;font-size:0.82rem;font-weight:600;border:1px solid #c8e6c9'>{meta['tag']}</span>
            </div>
            <div style='margin-top:6px;color:{INK_2};font-size:0.95rem;line-height:1.5'>
                <b>Kiya karti hai (Concept):</b> {meta['full_desc']} · Charges of {cost:.2f}% per side.
            </div>
            <div style='margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px'>
                <div style='background:#f1f8e9;padding:10px 14px;border-radius:6px;border:1px solid #c8e6c9'>
                    <b style='color:#2e7d32;font-size:0.95rem'>🟢 Kharidne ka Niyam (Entry Rule):</b>
                    <div style='font-size:0.9rem;color:{INK_1};margin-top:4px;line-height:1.4'>{meta.get('entry_rule', '')}</div>
                </div>
                <div style='background:#fbe9e7;padding:10px 14px;border-radius:6px;border:1px solid #ffccbc'>
                    <b style='color:#c62828;font-size:0.95rem'>🔴 Bechne ka Niyam (Exit Rule):</b>
                    <div style='font-size:0.9rem;color:{INK_1};margin-top:4px;line-height:1.4'>{meta.get('exit_rule', '')}</div>
                </div>
            </div>
            <div style='margin-top:10px;padding:8px 12px;background:#f8f9fa;border-radius:6px;font-size:0.88rem;color:{INK_2};line-height:1.5'>
                🎯 <b>Kab sabse accha kaam karegi:</b> {meta.get('best_market', '')}<br>
                ⚠️ <b>Khatra / Savdhani:</b> {meta.get('risk_watch', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    trades, skipped = strat.SIGNALS[selected_s_id](daily, asof, start, cost, opens=dp.load_open_data(stock_ticker))
    book, final_cash, value, bh_value, bh_shares = simulate(daily, trades, start, capital, cost)

    final, bh_final = float(value.iloc[-1]), float(bh_value.iloc[-1])
    pnl, bh_pnl = final - capital, bh_final - capital
    last_px = daily["Close"].iloc[-1]

    if book and book[0]["shares"] == 0:
        st.warning(f"{inr(capital)} is less than one share (₹{book[0]['buy_px'] * (1 + cost / 100):.2f} with charges). Enter a larger amount.")

    made = pnl >= 0
    status_box(UP_FG if made else DOWN_FG,
               f"{'PROFIT' if made else 'LOSS'} {inr(pnl, sign=True)}",
               [f"You put in <b>{inr(capital)}</b> on {start.strftime('%d %b %Y')} → worth <b>{inr(final)}</b> on {daily.index[-1].strftime('%d %b %Y')} ({pct(100 * pnl / capital)}).",
                f"Buying {bh_shares} shares on {start.strftime('%d %b %Y')} and holding would be worth <b>{inr(bh_final)}</b> ({inr(bh_pnl, sign=True)}, {pct(100 * bh_pnl / capital)}).",
                f"The strategy did <b>{'better' if final >= bh_final else 'worse'}</b> than buy & hold by {inr(abs(final - bh_final))}."])
    st.write("")

    closed = [b for b in book if b.get("sell") is not None]
    wins = [b for b in closed if b.get("pnl", 0.0) > 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Money now", inr(final), f"{inr(pnl, sign=True)} on {inr(capital)}")
    c2.metric("Total charges paid", inr(sum(b.get("charges", 0.0) for b in book)), f"{cost:.2f}% per buy/sell", delta_color="off")
    c3.metric("Trades won / lost", f"{len(wins)} / {len(closed) - len(wins)}",
              f"{len(closed)} closed{' + 1 open' if book and book[-1].get('sell') is None else ''}", delta_color="off")
    tot_held_days = sum(b.get("held", 0) for b in book)
    c4.metric("Money in the stock", f"{100 * tot_held_days / max(1, len(daily[daily.index >= start])):.0f}% of days",
              f"best {inr(max((b['pnl'] for b in closed), default=0), sign=True)} · worst {inr(min((b['pnl'] for b in closed), default=0), sign=True)}" if closed else "", delta_color="off")

    # Comprehensive P&L Attribution & Market Analysis Section (Why Profit / Loss Happened)
    pnl_info = strat.explain_pnl(book, capital, final, cost, stock_name, meta["name"])
    pnl_bg = "#f1f8e9" if pnl_info["is_profit"] else "#fff3e0"
    pnl_border = "#81c784" if pnl_info["is_profit"] else "#ffb74d"
    pnl_header_color = UP_FG if pnl_info["is_profit"] else DOWN_FG

    with st.expander(f"🔍 **Why Profit / Loss Happened (Kyo Profit ya Loss Hua?) — {pnl_info['headline']}**", expanded=True):
        st.markdown(
            f"""
            <div style='background:{pnl_bg};border-left:5px solid {pnl_header_color};border:1px solid {pnl_border};padding:12px 16px;border-radius:6px;margin-bottom:12px'>
                <div style='font-size:1.1rem;font-weight:700;color:{pnl_header_color}'>{pnl_info['headline']}</div>
                <div style='font-size:0.95rem;color:{INK_1};margin-top:4px'><b>English:</b> {pnl_info['summary_en']}</div>
                <div style='font-size:0.95rem;color:{INK_2};margin-top:4px'><b>Hindi:</b> {pnl_info['summary_hi']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Key Financial Attribution Drivers
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Gross Profit from Wins", inr(pnl_info["gross_profit"]), f"{pnl_info['winning_trades']} wins (avg {pct(pnl_info['avg_win_pct'])})", delta_color="normal")
        k2.metric("Gross Loss from Losers", f"-{inr(pnl_info['gross_loss'])}", f"{pnl_info['losing_trades']} losses (avg {pct(pnl_info['avg_loss_pct'])})", delta_color="inverse")
        k3.metric("Profit Factor", f"{pnl_info['profit_factor']:.2f}x", "Gross Win / Gross Loss", delta_color="off")
        k4.metric("Total Brokerage & Fees", inr(pnl_info["total_charges"]), f"{cost:.2f}% per order", delta_color="off")

        st.markdown(f"**Key Drivers & Market Behavior (Asal Wajah):**")
        for b_en, b_hi in zip(pnl_info["bullets_en"], pnl_info["bullets_hi"]):
            st.markdown(f"• **{b_en}**<br>&nbsp;&nbsp;↳ *Hindi:* {b_hi}", unsafe_allow_html=True)

    dq, wq = daily["Quadrant"].iloc[-1], asof["Quadrant"].iloc[-1]
    ok_sym = lambda q: "✅" if q in UP else "❌"
    state = [f"Weekly now: {ok_sym(wq)} <b style='color:{QUAD_FG[wq]}'>{wq}</b>  ·  Daily now: {ok_sym(dq)} <b style='color:{QUAD_FG[dq]}'>{dq}</b>"]
    if meta["weekly_only"]:   # weekly only: report the last completed week, never the daily
        w_now = asof.iloc[-1]
        state = [f"Weekly as of {asof.index[-1].strftime('%a %d %b %Y')} (week so far): {ok_sym(wq)} "
                 f"<b style='color:{QUAD_FG[wq]}'>{wq}</b> — Weekly RS-Ratio {w_now['RS_Ratio']:.2f}, "
                 f"RS-Momentum {w_now['RS_Momentum']:.2f}. Checked every trading day; the daily chart is not used."]
    open_b = book[-1] if book and book[-1].get("sell") is None else None
    st.write("")
    if open_b and open_b["buy"] == daily.index[-1]:
        status_box(QUAD_FG["Leading"], "▲ BUY signal today",
                   [f"Buy {open_b['shares']} shares at about ₹{open_b['buy_px']:.2f} ({inr(open_b['paid'])} with charges)."] + state)
    elif open_b:
        status_box(QUAD_FG["Leading"], f"HOLD — {open_b['shares']} shares since {open_b['buy'].strftime('%a %d %b %Y')}",
                   [f"Bought at ₹{open_b['buy_px']:.2f} (paid {inr(open_b['paid'])}). Today ₹{last_px:.2f} — selling now would give {inr(open_b['got'])}: {inr(open_b['pnl'], sign=True)}.",
                    (f"Sell rule: {meta['exit_rule']}"
                     if meta["weekly_only"] else
                     f"Current status: {open_b.get('exit_reason', 'Open Position')}. Daily RS-Momentum: {daily['RS_Momentum'].iloc[-1]:.2f}.")] + state)
    elif closed and closed[-1]["sell"] == daily.index[-1]:
        b = closed[-1]
        status_box(QUAD_FG["Lagging"], "▼ SELL signal today",
                   [f"Sell all {b['shares']} shares at about ₹{b['sell_px']:.2f} → {inr(b['got'])} ({inr(b['pnl'], sign=True)}). Reason: {b.get('exit_reason', '')}"] + state)
    else:
        status_box(INK_3, f"WAIT — all {inr(final_cash)} is in cash",
                   [f"Awaiting next entry signal for {meta['name']}."] + state)

    st.plotly_chart(strategy_figure(daily, asof, book, value, bh_value, start, capital, stock_name),
                    width="stretch", theme=None, config={"displaylogo": False})

    st.subheader("Every trade in rupees, oldest first")
    if book:
        table = pd.DataFrame([{
            "#": i + 1,
            "Buy date": b["buy"].strftime("%d %b %Y"),
            "Buy price": f"₹{b['buy_px']:.2f}",
            "Shares": b["shares"],
            "You paid": inr(b["paid"]),
            "Sell date": b["sell"].strftime("%d %b %Y") if b["sell"] is not None else "still holding",
            "Sell price": f"₹{b['sell_px']:.2f}" + ("" if b["sell"] is not None else " (today)"),
            "You got": inr(b["got"]) + ("" if b["sell"] is not None else " (if sold)"),
            "Profit / loss": inr(b["pnl"], sign=True),
            "%": pct(b["pnl_pct"]),
            "Outcome / Reason (Kyo Profit ya Loss)": strat.trade_outcome_reason(b, cost),
            "Charges": inr(b["charges"]),
            "Exit Reason": b.get("exit_reason", ""),
            "Money after": inr(b["cash_after"]),
        } for i, b in enumerate(book)])

        def money_style(col):
            return ["color: %s; font-weight: bold" % (UP_FG if v.startswith("+") else
                                                      DOWN_FG if v.startswith("-") else INK_2) for v in col]

        st.dataframe(table.style.apply(money_style, subset=["Profit / loss", "%"]),
                     hide_index=True, width="stretch", height=38 + 35 * min(15, len(table)))
        clean_name = dp.sanitize_ticker(stock_ticker).lower()
        full = pd.DataFrame([{
            "strategy": meta["name"],
            "buy_date": b["buy"].date(), "buy_price": round(b["buy_px"], 2), "shares": b["shares"],
            "paid": round(b["paid"], 2), "weekly_at_buy": b.get("weekly", ""),
            "daily_move": f"{b.get('frm', '')} -> {b.get('into', '')}",
            "daily_at_sell": b.get("daily_at_sell", ""), "weekly_at_sell": b.get("weekly_at_sell", ""),
            "sell_date": b["sell"].date() if b["sell"] is not None else "open",
            "sell_price": round(b["sell_px"], 2), "got": round(b["got"], 2),
            "profit_loss": round(b["pnl"], 2), "charges": round(b["charges"], 2),
            "exit_reason": b.get("exit_reason", ""),
            "outcome_reason": strat.trade_outcome_reason(b, cost),
            "money_after": round(b["cash_after"], 2),
        } for b in book])
        st.download_button("Download trades as CSV", full.to_csv(index=False),
                           file_name=f"{clean_name}_strategy_{selected_s_id}_trades.csv", mime="text/csv",
                           key=f"dl_trades_{stock_ticker}_{selected_s_id}")
    else:
        st.info(f"No buy signal since {start.strftime('%d %b %Y')}, so money stayed in cash.")

    if skipped:
        with st.expander(f"Daily turned up but weekly was down — no buy ({len(skipped)} times)"):
            st.dataframe(pd.DataFrame([{"Date": s["date"].strftime("%a %d %b %Y"),
                                        "Daily moved": f"{s['frm']} → {s['into']}",
                                        "Weekly that day": s["weekly"]} for s in skipped])
                         .style.apply(quad_style, subset=["Weekly that day"]),
                         hide_index=True, width="stretch")


# ---- Sidebar & Main Execution -----------------------------------------------
with st.sidebar:
    # tags = listing year (agreed with the user); the stock list and the All stocks page follow the tag
    st.header("Stock & View Selection")
    selected_tag = st.selectbox("Tag", cfg.TAGS, key="tag",
                                format_func=lambda g: f"{g} ({len(cfg.stocks_for_tag(g))} stocks)")
    tag_stocks = cfg.stocks_for_tag(selected_tag)
    selected_stock = st.selectbox(
        "Select Stock",
        options=tag_stocks,
        format_func=lambda t: f"{cfg.STOCKS[t]['name']} ({t[:-3]} · listed {pd.Timestamp(cfg.STOCKS[t]['listing_date']):%d %b %Y})",
        index=0, key="stock"
    )
    stock_meta = cfg.STOCKS[selected_stock]
    st.caption(f"**Selected:** {stock_meta['name']} · Sector: *{stock_meta['sector'] or '—'}* · "
               f"issue price ₹{stock_meta['issue_price']:,.2f}")
    if cfg.WAITING:
        st.caption(f"{len(cfg.WAITING)} newer IPOs have fewer than {cfg.MIN_HISTORY_ROWS} sessions: they are on the "
                   f"IPO details page and join these charts automatically once they have enough data.")

    page_choice = st.radio("Timeframe / Page", ["Daily", "Weekly", "Both", "All stocks"],
                           horizontal=False, key="page",
                           help="Daily is the main view for IPO stocks; Weekly needs about 21 months of listing history.")

    st.markdown("---")
    if st.button("🔄 Refresh All Data", help=f"Add the latest NSE sessions for all {len(cfg.ALL_STOCKS)} IPO stocks + NIFTY 50 from NSE's official files"):
        with st.spinner("Downloading NSE files..."):
            res = dp.refresh_data(force=True)
            if res["status"] in ("SUCCESS", "NO_NEW_SESSION"):
                st.success(res.get("message"))
                st.rerun()
            else:
                st.error(f"Refresh failed: {res.get('message')}")

    if page_choice == "Strategy":
        strategy_choice = st.selectbox(
            "Strategy", options=list(strat.STRATEGY_REGISTRY),
            format_func=lambda i: f"{strat.STRATEGY_REGISTRY[i]['label']}: {strat.STRATEGY_REGISTRY[i]['name']}",
            key="strategy_choice")
        st.caption(f"💡 *{strat.STRATEGY_REGISTRY[strategy_choice]['short_desc']}*")
        with st.form("strategy_inputs"):
            capital = st.number_input("Amount to invest (₹)", min_value=1000, max_value=10**9,
                                      value=10000, step=1000, format="%d")
            bt_from = st.date_input("Start date", value=CYCLES_FROM.date(),
                                    min_value=datetime(2010, 1, 1).date(), key="bt_from")
            cost = st.number_input("Charges per buy / sell (%)", min_value=0.0, max_value=2.0,
                                   value=0.12, step=0.01, format="%.2f")
            st.form_submit_button("▶ Run strategy", type="primary", width="stretch")
    elif page_choice == "All stocks":
        all_tf = st.radio("Comparison Timeframe", ["Daily", "Weekly"], horizontal=True)
    else:
        active_tf = "Daily" if page_choice == "Both" else page_choice
        cfg_tf = TF_CONFIG[active_tf]
        window = st.select_slider("Slider covers", options=[SINCE_2024] + cfg_tf["windows"],
                                  value=SINCE_2024, key=f"win_{page_choice}",
                                  format_func=lambda v: v if v == SINCE_2024 else f"last {v} {cfg_tf['units']}")
        if page_choice == "Both":
            tail_n = st.slider("Daily trail (sessions)", 5, 60, TF_CONFIG["Daily"]["tail"], key="tail_d")
            tail_w = st.slider("Weekly trail (weeks)", 5, 60, TF_CONFIG["Weekly"]["tail"], key="tail_w")
        else:
            tail_n = st.slider(f"Trail length ({cfg_tf['units']})", 5, 60, cfg_tf["tail"], key="tail_s")
        fixed_axes = st.toggle("Fixed axes while sliding", value=False)


# Main layout
render_status_panel(selected_stock)

if page_choice == "Strategy":
    strategy_page(selected_stock, strategy_choice)
elif page_choice == "All stocks":
    all_stocks_page(all_tf, tag_stocks, selected_tag)
elif page_choice == "Both":
    both_page(selected_stock)
else:
    single_page(selected_stock, page_choice)

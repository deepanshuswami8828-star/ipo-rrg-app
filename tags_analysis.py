"""
Tags analysis tab: profitability of Strategy 2 version 3 for every stock in a tag, over two periods,
and how it changes when a take-profit threshold is added.
Runs as its own page (top navigation in rrg_app.py), separate from the RRG chart pages.
Everything is driven by config.TAGS / config.stocks_for_tag, so new tags appear here automatically.
"""
import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import data_pipeline as dp
import strategies as strat

# one colour per period in the take-profit chart (solid = with take-profit, dashed = base strategy)
PERIOD_COLORS = ["#2a78d6", "#e65100", "#2e7d32", "#7b1fa2"]


# ---- data (cached per data version and strategy code) ---------------------------------------
def cache_key() -> str:
    """Data version + fingerprint of strategies.py. st.cache_data only hashes the cached function's own code,
    so without the fingerprint a change in strategies.py would keep serving results built by the old code."""
    with open(strat.__file__, "rb") as f:
        code = hashlib.sha256(f.read()).hexdigest()[:12]
    return f"{dp.load_manifest().get('version_id', 'none')}-{code}"


@st.cache_resource(show_spinner=False)
def ticker_inputs(ticker: str, manifest_version: str):
    """(first data date, Strategy 2 v3 inputs from the start of Period 1 or None) for one ticker.
    The inputs are built once and reused for every stop-loss / take-profit setting."""
    stock_df = dp.load_clean_data(ticker)
    bench_df = dp.load_clean_data(cfg.BENCHMARK_TICKER)
    if stock_df is None or bench_df is None or len(stock_df) == 0:
        return None, None
    name = cfg.STOCKS[ticker]["name"]
    aligned = dp.align_stock_with_benchmark(stock_df, bench_df, name, cfg.BENCHMARK_NAME)
    if aligned is None or len(aligned) == 0:
        return None, None
    daily, _ = strat.compute_rrg(aligned, name, tf="Daily")
    asof = strat.weekly_asof(aligned, name)
    if len(daily) < 2 or asof.empty:
        return aligned.index[0], None
    start = max(strat.TAG_PERIODS[0][1], daily.index[1], asof.index[0])
    return aligned.index[0], strat.v3_inputs(daily, asof, start, dp.load_high_data(ticker), dp.load_open_data(ticker))


def tag_members(tag: str, manifest_version: str):
    """(qualifying tickers, left-out tickers with their first date) for a tag."""
    firsts = {t: ticker_inputs(t, manifest_version)[0] for t in cfg.stocks_for_tag(tag)}
    qualified = [t for t, f in firsts.items() if f is not None and f <= strat.TAG_DATA_FROM]
    return qualified, {t: f for t, f in firsts.items() if t not in qualified}


@st.cache_data(show_spinner=False)
def tag_reports(tag: str, manifest_version: str, take_profit_pct=None, stop_loss_pct=strat.STOP_LOSS_V3):
    """{period label: strat.period_report} for a tag: base strategy or with a take-profit threshold,
    at a given stop-loss level."""
    qualified, _ = tag_members(tag, manifest_version)
    trades = {}
    for t in qualified:
        inp = ticker_inputs(t, manifest_version)[1]
        trades[t] = strat.v3_trades(inp, strat.TAG_COST_PCT, take_profit_pct, stop_loss_pct)[0] if inp else []
    return {label: strat.period_report(trades, p_start, p_end) for label, p_start, p_end in strat.TAG_PERIODS}


# ---- page ------------------------------------------------------------------------------------
st.title("🏷️ Tags analysis")
st.caption("Profitability of Strategy 2 · Version 3 (weekly RRG, 10% stop-loss) for every stock in a tag.")

tag = st.selectbox("Tag", cfg.TAGS, key="tags_page_tag",
                   format_func=lambda g: f"{g} ({len(cfg.stocks_for_tag(g))} stocks)")
st.markdown(f"### Tag: {tag}")
st.caption(f"Strategy run from {strat.TAG_PERIODS[0][1]:%d %b %Y}. {strat.ENTRY_TEXT} {strat.RRG_EXIT_TEXT} Each trade assumes ₹{strat.TAG_TRADE_AMOUNT:,.0f} invested, "
           f"charges {strat.TAG_COST_PCT:g}% per side. A trade counts in a period only if it was bought and sold inside it "
           f"(a trade spanning both periods, or still open, is left out). A ticker counts only if its data starts by April 2022.")

m_ver = cache_key()
with st.spinner("Running the strategy for every stock in the tag..."):
    qualified, left_out = tag_members(tag, m_ver)
    base = tag_reports(tag, m_ver)
if left_out:
    st.caption("Not included (data starts after April 2022): " + ", ".join(
        f"{t.replace('.NS', '')} (from {f:%d %b %Y})" if f is not None else t.replace(".NS", "") for t, f in left_out.items()))

for label, p_start, p_end in strat.TAG_PERIODS:
    rep = base[label]
    rows = [{
        "Ticker": t.replace(".NS", ""),
        "Number of trades": s["trades"],
        "Profit % per trade": ", ".join(f"{p:.2f}" for p in s["profits"]) or "—",
        "Days held per trade": ", ".join(str(x) for x in s["days"]) or "—",
        "Sum of profit %": round(s["profit_sum"], 2),
        "Sum of days held": s["days_sum"],
        "Daily profit %": round(s["daily_profit"], 4) if s["daily_profit"] is not None else None,
    } for t, s in rep["rows"].items()]
    avg = rep["avg_daily_profit"]

    st.markdown(f"#### {label}: {p_start:%b %Y} to {p_end:%b %Y}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers in period", rep["tickers"])
    c2.metric("Closed trades in period", rep["trades"])
    c3.metric("Average daily profit % (simple)", f"{avg:.4f}%" if avg is not None else "—",
              help="Simple average of 'Daily profit %' over the tickers that had at least one closed trade.")
    if len(rows) > strat.TAG_MAX_ROWS:
        st.caption(f"Showing {strat.TAG_MAX_ROWS} of {len(rows)} tickers.")
    shown = rows[:strat.TAG_MAX_ROWS]
    st.dataframe(pd.DataFrame(shown), hide_index=True, width="stretch", height=38 + 35 * max(1, len(shown)),
                 column_config={"Profit % per trade": st.column_config.TextColumn(width="large"),
                                "Days held per trade": st.column_config.TextColumn(width="medium"),
                                "Daily profit %": st.column_config.NumberColumn(format="%.4f"),
                                "Sum of profit %": st.column_config.NumberColumn(format="%.2f")})
    if rep["tickers_with_trades"] < rep["tickers"]:
        st.caption(f"{rep['tickers'] - rep['tickers_with_trades']} ticker(s) had no closed trade in this period, "
                   f"so they have no daily profit % and are left out of the average.")

# ---- take-profit threshold analysis, one compact chart per stop-loss level ----------------------
def threshold_chart(stop_loss: float, base_rep: dict, curve_rep: dict, y_range: list) -> go.Figure:
    """Take-profit threshold (x) vs simple average of daily profit % (y), one line per period;
    dashed line per period = base strategy (no take-profit) at the same stop-loss."""
    fig = go.Figure()
    for color, (label, p_start, p_end) in zip(PERIOD_COLORS, strat.TAG_PERIODS):
        pts = [(x, curve_rep[x][label]["avg_daily_profit"], curve_rep[x][label]["avg_days_held"], curve_rep[x][label]["trades"])
               for x in strat.TAKE_PROFIT_THRESHOLDS if curve_rep[x][label]["avg_daily_profit"] is not None]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], customdata=[[p[2], p[3]] for p in pts],
                                 mode="lines+markers", name=f"{label} ({p_start:%b %y}-{p_end:%b %y})",
                                 line=dict(color=color, width=2), marker=dict(size=4),
                                 hovertemplate=f"<b>{label} · {stop_loss:g}% stop-loss</b><br>Take-profit %{{x}}%<br>"
                                               "Avg daily profit %{y:.4f}%<br>Avg days held per trade %{customdata[0]:.1f}<br>"
                                               "Closed trades %{customdata[1]}<extra></extra>"))
        b = base_rep[label]
        if b["avg_daily_profit"] is not None:
            fig.add_trace(go.Scatter(x=[0, 100], y=[b["avg_daily_profit"]] * 2, mode="lines",
                                     name=f"{label} base: {b['avg_daily_profit']:.4f}%",
                                     line=dict(color=color, width=1.5, dash="dash"),
                                     customdata=[[b["avg_days_held"], b["trades"]]] * 2,
                                     hovertemplate=f"<b>{label} · {stop_loss:g}% stop-loss · base (no take-profit)</b><br>"
                                                   "Avg daily profit %{y:.4f}%<br>Avg days held per trade %{customdata[0]:.1f}<br>"
                                                   "Closed trades %{customdata[1]}<extra></extra>"))
    fig.update_layout(title=dict(text=f"{stop_loss:g}% stop-loss", font=dict(size=15)), height=380,
                      margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="Take-profit threshold x (%)", yaxis_title="Avg daily profit %",
                      xaxis=dict(range=[0, 102], dtick=10), yaxis=dict(range=y_range),
                      legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
                      plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb")
    return fig


st.markdown("---")
st.markdown("### Take-profit threshold analysis by stop-loss")
st.caption("Modified Strategy 2 · Version 3: on any day after the buy, if the day's high reaches the buy price + x%, the stock "
           "is sold that day at exactly buy price + x% (x% profit booked). After such an exit it is bought again only once "
           "the weekly has dropped into Weakening/Lagging and come back up, as after a stop-loss. Each chart uses one "
           "stop-loss level; everything else (weekly exits, periods, ₹1,00,000 per trade) is unchanged. One colour per "
           "period: solid line = with take-profit, dashed line = base strategy without take-profit at the same stop-loss. "
           "All charts share the same y-axis so they can be compared. Hover a point for the average days held per trade.")

with st.spinner(f"Testing {len(strat.TAKE_PROFIT_THRESHOLDS)} take-profit thresholds x {len(strat.TAG_STOP_LOSSES)} "
                "stop-loss levels for every stock in the tag..."):
    by_sl = {sl: (tag_reports(tag, m_ver, None, sl),
                  {x: tag_reports(tag, m_ver, float(x), sl) for x in strat.TAKE_PROFIT_THRESHOLDS})
             for sl in strat.TAG_STOP_LOSSES}

values = [r[label]["avg_daily_profit"] for b, c in by_sl.values() for r in [b] + list(c.values())
          for label, _, _ in strat.TAG_PERIODS if r[label]["avg_daily_profit"] is not None]
pad = (max(values) - min(values)) * 0.05 if values else 0.01
y_range = [min(values) - pad, max(values) + pad] if values else None

for row_start in range(0, len(strat.TAG_STOP_LOSSES), 2):
    cols = st.columns(2)
    for col, sl in zip(cols, strat.TAG_STOP_LOSSES[row_start:row_start + 2]):
        base_rep, curve_rep = by_sl[sl]
        col.plotly_chart(threshold_chart(sl, base_rep, curve_rep, y_range), width="stretch", key=f"tp_chart_{sl:g}")

with st.expander("Numbers behind the charts"):
    for tab, sl in zip(st.tabs([f"{sl:g}% stop-loss" for sl in strat.TAG_STOP_LOSSES]), strat.TAG_STOP_LOSSES):
        base_rep, curve_rep = by_sl[sl]
        table = []
        for x in [None] + strat.TAKE_PROFIT_THRESHOLDS:
            reps = base_rep if x is None else curve_rep[x]
            row = {"Take-profit threshold": "Base (none)" if x is None else f"{x}%"}
            for label, _, _ in strat.TAG_PERIODS:
                a, d = reps[label]["avg_daily_profit"], reps[label]["avg_days_held"]
                row[f"{label}: avg daily profit %"] = round(a, 4) if a is not None else None
                row[f"{label}: closed trades"] = reps[label]["trades"]
                row[f"{label}: avg days held"] = round(d, 1) if d is not None else None
            table.append(row)
        tab.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

"""
RRG calculation for the IPO RRG app, unchanged from the RRG app (make_rrg_pvrinox.py / strategies.py):

    rs          = stock close / NIFTY 50 close
    rs_smooth   = SMA(rs, 5)
    RS-Ratio    = 100 * EMA(rs_smooth, 5) / EMA(rs_smooth, 30)
    RS-Momentum = 100 * EMA(RS-Ratio, 3) / EMA(RS-Ratio, 14)        (EMAs with adjust=False)

Quadrants: Leading (ratio >= 100, momentum >= 100), Weakening (>= 100, < 100), Lagging (< 100, < 100),
Improving (< 100, >= 100). Weekly = the last trading day of each week (W-FRI). The first bars are a warm-up:
min(WARMUP, bars // 3) bars are dropped, as in the RRG app.
"""
from typing import Tuple

import pandas as pd

import config as cfg

RS_PRESMOOTH, RS_SHORT, RS_LONG, MOM_SHORT, MOM_LONG = 5, 5, 30, 3, 14
ORDER = ["Improving", "Leading", "Weakening", "Lagging"]      # clockwise rotation

# palette of the RRG app
SURFACE, INK_1, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e7e6e2"
QUAD_BG = {"Leading": "#edf4ee", "Weakening": "#f7f3e8", "Lagging": "#f8eeee", "Improving": "#ebf1f8"}
QUAD_FG = {"Leading": "#3f7d55", "Weakening": "#8a6d2f", "Lagging": "#a05252", "Improving": "#4a6c96"}


def calc_rs_ratio(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    rs = stock_close / bench_close
    rs_smooth = rs.rolling(RS_PRESMOOTH, min_periods=1).mean()
    return 100.0 * rs_smooth.ewm(span=RS_SHORT, adjust=False).mean() / rs_smooth.ewm(span=RS_LONG, adjust=False).mean()


def calc_rs_momentum(rs_ratio: pd.Series) -> pd.Series:
    return 100.0 * rs_ratio.ewm(span=MOM_SHORT, adjust=False).mean() / rs_ratio.ewm(span=MOM_LONG, adjust=False).mean()


def quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100:
        return "Weakening"
    if rs_momentum < 100:
        return "Lagging"
    return "Improving"


def compute_rrg(aligned: pd.DataFrame, stock_name: str, tf: str = "Daily") -> Tuple[pd.DataFrame, bool]:
    """RRG series (RS_Ratio, RS_Momentum, Close, Quadrant) from closes aligned with NIFTY 50.
    Returns (frame, sufficient): sufficient = at least 30 daily bars, or cfg.MIN_WEEKLY_BARS weekly bars."""
    closes = aligned.copy()
    if tf == "Weekly":
        closes = closes.groupby(closes.index.to_period("W-FRI")).tail(1)
    warmup = min(cfg.WARMUP_WEEKLY if tf == "Weekly" else cfg.WARMUP_DAILY, len(closes) // 3)
    if len(closes) < warmup + 10:
        return pd.DataFrame(), False
    rr = calc_rs_ratio(closes[stock_name], closes[cfg.BENCHMARK_NAME])
    rm = calc_rs_momentum(rr)
    df = pd.DataFrame({"RS_Ratio": rr, "RS_Momentum": rm, "Close": closes[stock_name]}).dropna().iloc[warmup:]
    df["Quadrant"] = [quadrant(a, b) for a, b in zip(df["RS_Ratio"], df["RS_Momentum"])]
    return df, len(df) >= (cfg.MIN_WEEKLY_BARS if tf == "Weekly" else 30)

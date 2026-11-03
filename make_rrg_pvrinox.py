"""
Relative Rotation Graph (RRG): PVR INOX vs NIFTY 50  -  daily.

Calculation is the calibrated EMA method from rrg_logic.py, unchanged:

    Step 1  rs        = pvrinox_close / nifty50_close
    Step 2  rs_smooth = SMA(rs, RS_PRESMOOTH)
    Step 3  RS-Ratio    = 100 * EMA(rs_smooth, 5) / EMA(rs_smooth, 30)
    Step 4  RS-Momentum = 100 * EMA(RS-Ratio, 3) / EMA(RS-Ratio, 14)

Quadrants (read clockwise):
    Improving (top-left) -> Leading (top-right) -> Weakening (bottom-right)
    -> Lagging (bottom-left)

Only TAIL_DAYS differs from rrg_logic.py's value (5): a single-name chart carries
one trail instead of sixteen, so a longer tail stays legible and shows the
rotation. Prices are auto-adjusted daily closes from Yahoo Finance, with the
local CSVs (through 2026-08-13) as an offline fallback. data/ is not modified
apart from a new rrg_pvrinox_data.csv.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.abspath(os.path.dirname(__file__))
CHARTS = os.path.join(ROOT, "charts")

# ---- method parameters (from rrg_logic.py; printed with the output) ---------
RS_PRESMOOTH = 5     # SMA window on raw RS before the double EMA
RS_SHORT = 5         # fast EMA for RS-Ratio
RS_LONG = 30         # slow EMA for RS-Ratio (baseline)
MOM_SHORT = 3        # fast EMA for RS-Momentum
MOM_LONG = 14        # slow EMA for RS-Momentum
TAIL_DAYS = 25       # trading days of trajectory drawn (~5 weeks)
PERIOD = "2y"        # download window; EMA(30) needs a long warm-up

STOCK, STOCK_TICKER = "PVR INOX", "PVRINOX.NS"
BENCH, BENCH_TICKER = "NIFTY 50", "^NSEI"

# ---- palette: same tokens as the project's existing charts ------------------
SURFACE, INK_1, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e7e6e2"
SERIES = "#2a78d6"                    # colour follows the entity, not its quadrant
QUAD_BG = {"Leading": "#edf4ee", "Weakening": "#f7f3e8",
           "Lagging": "#f8eeee", "Improving": "#ebf1f8"}
QUAD_FG = {"Leading": "#3f7d55", "Weakening": "#8a6d2f",
           "Lagging": "#a05252", "Improving": "#4a6c96"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK_1,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
})


# ---- calculation (verbatim from rrg_logic.py) -------------------------------
def calc_rs_ratio(sector_close, benchmark_close):
    rs = sector_close / benchmark_close
    rs_smooth = rs.rolling(RS_PRESMOOTH, min_periods=1).mean()
    ema_fast = rs_smooth.ewm(span=RS_SHORT, adjust=False).mean()
    ema_slow = rs_smooth.ewm(span=RS_LONG, adjust=False).mean()
    return 100.0 * ema_fast / ema_slow


def calc_rs_momentum(rs_ratio):
    mom_fast = rs_ratio.ewm(span=MOM_SHORT, adjust=False).mean()
    mom_slow = rs_ratio.ewm(span=MOM_LONG, adjust=False).mean()
    return 100.0 * mom_fast / mom_slow


def quadrant(rs_ratio, rs_momentum):
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "Lagging"
    return "Improving"


# ---- data -------------------------------------------------------------------
def load_closes():
    """Daily closes for the stock and the benchmark, aligned. -> (frame, source)."""
    try:
        import data_pipeline as dp
        pvr_clean = dp.load_clean_data("PVRINOX.NS")
        bench_clean = dp.load_clean_data(BENCH_TICKER)
        if pvr_clean is not None and bench_clean is not None:
            aligned = dp.align_stock_with_benchmark(pvr_clean, bench_clean, STOCK, BENCH)
            if len(aligned) >= RS_LONG + TAIL_DAYS:
                return aligned, "Verified local clean data"
    except Exception:
        pass

    stock = pd.read_csv(os.path.join(ROOT, "pvrinox_ticker_2022-04-01_to_2026-08-13.csv"),
                        parse_dates=["Date"], index_col="Date")
    col = "Adj Close" if "Adj Close" in stock.columns else "Close"
    stock_s = stock[col]
    bench = pd.read_csv(os.path.join(ROOT, "nifty50_dataset_2024-04-01_to_2026-08-13.csv"),
                        parse_dates=["Date"], index_col="Date")["Close"]
    # Strict inner join, no ffill, no bfill, no interpolation
    closes = pd.concat([stock_s.rename(STOCK), bench.rename(BENCH)], axis=1, join="inner").dropna()
    return closes, "local CSVs in this repo"


# ---- plot -------------------------------------------------------------------
def plot(df, source, out_path):
    tail = df.tail(TAIL_DAYS)
    xs, ys = tail["RS_Ratio"].to_numpy(), tail["RS_Momentum"].to_numpy()
    n = len(xs)
    here = quadrant(xs[-1], ys[-1])

    # frame the trail, but always keep the 100/100 crosshair in view with a
    # visible margin on the far side so the quadrant it sits against is readable
    spread = max(xs.max() - xs.min(), ys.max() - ys.min())
    pad = max(0.35, 0.18 * spread)
    margin = max(0.45, 0.25 * spread)
    x_lo = min(xs.min() - pad, 100 - margin)
    x_hi = max(xs.max() + pad, 100 + margin)
    y_lo = min(ys.min() - pad, 100 - margin)
    y_hi = max(ys.max() + pad, 100 + margin)
    xr, yr = x_hi - x_lo, y_hi - y_lo

    fig, ax = plt.subplots(figsize=(10, 8.6))

    ax.fill_between([x_lo, 100], 100, y_hi, color=QUAD_BG["Improving"], zorder=0)
    ax.fill_between([100, x_hi], 100, y_hi, color=QUAD_BG["Leading"], zorder=0)
    ax.fill_between([x_lo, 100], y_lo, 100, color=QUAD_BG["Lagging"], zorder=0)
    ax.fill_between([100, x_hi], y_lo, 100, color=QUAD_BG["Weakening"], zorder=0)

    lkw = dict(fontsize=10.5, fontweight="bold", alpha=0.9)
    ax.text(x_lo + 0.04 * xr, y_hi - 0.04 * yr, "IMPROVING",
            color=QUAD_FG["Improving"], va="top", ha="left", **lkw)
    ax.text(x_hi - 0.04 * xr, y_hi - 0.04 * yr, "LEADING",
            color=QUAD_FG["Leading"], va="top", ha="right", **lkw)
    ax.text(x_lo + 0.04 * xr, y_lo + 0.04 * yr, "LAGGING",
            color=QUAD_FG["Lagging"], va="bottom", ha="left", **lkw)
    ax.text(x_hi - 0.04 * xr, y_lo + 0.04 * yr, "WEAKENING",
            color=QUAD_FG["Weakening"], va="bottom", ha="right", **lkw)

    ax.axvline(100, color=INK_3, linewidth=1.1, zorder=1)
    ax.axhline(100, color=INK_3, linewidth=1.1, zorder=1)
    ax.grid(True, linestyle=":", linewidth=0.5, color=GRID, zorder=0)
    ax.set_axisbelow(True)

    # trail: older points recede, today's point is solid and ringed in the surface
    for i in range(n - 1):
        a = 0.22 + 0.78 * (i / (n - 2)) if n > 2 else 1.0
        ax.plot(xs[i:i + 2], ys[i:i + 2], "-", color=SERIES, linewidth=2,
                alpha=a, solid_capstyle="round", zorder=2)
        ax.scatter(xs[i], ys[i], s=26, color=SERIES, alpha=a, linewidths=0, zorder=3)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=SERIES, lw=2))
    ax.scatter(xs[-1], ys[-1], s=140, color=SERIES, edgecolors=SURFACE,
               linewidths=2, zorder=5)

    # date labels only where they carry information: trail start and today.
    # Flip a label to the inside when its point sits near the right edge.
    def side(x, gap):
        return (-gap, "right") if (x - x_lo) / xr > 0.72 else (gap, "left")

    dx, ha = side(xs[0], 9)
    ax.annotate(tail.index[0].strftime("%d %b"), (xs[0], ys[0]),
                textcoords="offset points", xytext=(dx, -4), ha=ha,
                fontsize=9, color=INK_3, va="center", zorder=6)
    dx, ha = side(xs[-1], 14)
    ax.annotate("%s  -  %s" % (STOCK, tail.index[-1].strftime("%d %b %Y")),
                (xs[-1], ys[-1]), textcoords="offset points", xytext=(dx, 0), ha=ha,
                fontsize=10.5, fontweight="bold", color=INK_1, va="center", zorder=6)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("JdK RS-Ratio  -  relative strength level  (100 = in line with NIFTY 50)",
                  fontsize=10)
    ax.set_ylabel("JdK RS-Momentum  -  rate of change of relative strength", fontsize=10)
    ax.tick_params(labelsize=9)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(GRID)

    fig.text(0.012, 0.975, "%s against %s  -  currently %s" % (STOCK, BENCH, here.upper()),
             fontsize=15, fontweight="bold", color=INK_1, va="top")
    fig.text(0.012, 0.936,
             "Daily RRG, %d-day tail  ·  %s to %s"
             % (TAIL_DAYS, tail.index[0].strftime("%d %b %Y"),
                tail.index[-1].strftime("%d %b %Y")),
             fontsize=10.5, color=INK_2, va="top")
    fig.text(0.012, 0.032,
             "RS-Ratio = 100 x EMA(SMA(RS,%d),%d) / EMA(SMA(RS,%d),%d)   ·   "
             "RS-Momentum = 100 x EMA(RS-Ratio,%d) / EMA(RS-Ratio,%d)"
             % (RS_PRESMOOTH, RS_SHORT, RS_PRESMOOTH, RS_LONG, MOM_SHORT, MOM_LONG),
             fontsize=8.5, color=INK_3, va="bottom")
    fig.text(0.012, 0.008, "Source: %s" % source,
             fontsize=8.5, color=INK_3, va="bottom")

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.915))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(CHARTS, exist_ok=True)
    print("RRG - %s vs %s (daily)" % (STOCK, BENCH))
    closes, source = load_closes()
    print("  source: %s" % source)
    print("  %d aligned sessions, %s to %s"
          % (len(closes), closes.index[0].strftime("%d %b %Y"),
             closes.index[-1].strftime("%d %b %Y")))

    rr = calc_rs_ratio(closes[STOCK], closes[BENCH])
    rm = calc_rs_momentum(rr)
    df = pd.DataFrame({"RS_Ratio": rr, "RS_Momentum": rm}).dropna()
    # drop the EMA warm-up so the tail is not distorted by initialization
    df = df.iloc[RS_LONG + MOM_LONG:]
    df["Quadrant"] = [quadrant(a, b) for a, b in zip(df["RS_Ratio"], df["RS_Momentum"])]

    out_png = os.path.join(CHARTS, "rrg_pvrinox_vs_nifty50.png")
    plot(df, source, out_png)

    out_csv = os.path.join(ROOT, "data", "rrg_pvrinox_data.csv")
    df.tail(TAIL_DAYS).to_csv(out_csv, float_format="%.4f")

    print("  parameters: presmooth SMA %d, RS EMA %d/%d, momentum EMA %d/%d, tail %d days"
          % (RS_PRESMOOTH, RS_SHORT, RS_LONG, MOM_SHORT, MOM_LONG, TAIL_DAYS))
    print("\n  last 5 sessions")
    print(df.tail(5).to_string(float_format=lambda v: "%8.2f" % v))
    last = df.iloc[-1]
    print("\n  Current position: RS-Ratio %.2f, RS-Momentum %.2f  ->  %s"
          % (last["RS_Ratio"], last["RS_Momentum"], last["Quadrant"].upper()))
    print("  Quadrants over the tail: %s"
          % df["Quadrant"].tail(TAIL_DAYS).value_counts().to_dict())
    print("\n  chart -> %s" % os.path.relpath(out_png, ROOT))
    print("  data  -> %s" % os.path.relpath(out_csv, ROOT))


if __name__ == "__main__":
    main()

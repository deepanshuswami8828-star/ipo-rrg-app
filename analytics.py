"""
Per-stock summary of the IPO RRG app (no Streamlit), saved as data/summary.csv so the app opens fast.
Built by tools/build_app_data.py and again at the end of every data refresh.

For every IPO stock (ready and waiting): listing performance (issue -> listing day -> latest close, split/bonus
adjusted), subscription, sector, and for ready stocks the Daily RRG now (quadrant, since when), the first and the
latest turn to Improving after listing with the gain since, and the Weekly RRG quadrant when there is enough history.
"""
import os

import pandas as pd

import config as cfg
import data_pipeline as dp
import rrg_calc as rrg



def summary_path() -> str:
    """data/summary.csv of the data folder in use (worked out on every call: tests point cfg at a copy)."""
    return os.path.join(cfg.DATA_DIR, "summary.csv")


def stock_rrg(ticker: str, tf: str = "Daily"):
    """(RRG frame, sufficient) of a stock against NIFTY 50; (empty, False) without data."""
    s = cfg.ALL_STOCKS.get(ticker) or cfg.load_all_stocks().get(ticker, {})
    f, bench = dp.load_clean_data(ticker), dp.load_clean_data(cfg.BENCHMARK_TICKER)
    if f is None or bench is None:
        return pd.DataFrame(), False
    name = s.get("name", ticker)
    aligned = dp.align_stock_with_benchmark(f, bench, name)
    df, ok = rrg.compute_rrg(aligned, name, tf)
    return df, ok


def turns_to_improving(daily: pd.DataFrame) -> pd.Series:
    """Dates the Daily quadrant moved into Improving from another quadrant."""
    q = daily["Quadrant"]
    return q[(q == "Improving") & (q.shift(1) != "Improving") & q.shift(1).notna()]


def build_summary() -> pd.DataFrame:
    stocks = cfg.load_all_stocks()
    ipo_csv = os.path.join(cfg.DATA_DIR, "ipo_info.csv")
    info = pd.read_csv(ipo_csv).set_index("symbol") if os.path.exists(ipo_csv) else pd.DataFrame()
    rows = []
    for t, s in stocks.items():
        f = dp.load_price_frame(t)
        if f is None or not len(f):
            continue
        sym = s["symbol"]
        i = info.loc[sym] if sym in info.index else {}
        factor0 = float(f["RawClose"].iloc[0] / f["Close"].iloc[0])        # split/bonus since listing
        last = float(f["Close"].iloc[-1])
        issue_adj = s["issue_price"] / factor0
        row = {
            "ticker": t, "symbol": sym, "name": s["name"], "tag": s["tags"][0], "status": s["status"],
            "sector": s.get("sector", ""), "industry": s.get("industry", ""),
            "listing_date": s["listing_date"], "issue_price": s["issue_price"],
            "listing_close": i.get("listing_close"), "listing_gain_pct": i.get("listing_gain_close_pct"),
            "latest_date": str(f.index[-1].date()), "latest_close": round(float(f["RawClose"].iloc[-1]), 2),
            "gain_since_issue_pct": round((last / issue_adj - 1) * 100, 2),
            "split_bonus": round(factor0, 4) if abs(factor0 - 1) > 1e-6 else None,
            "sessions": int(len(f)),
            "sub_total_x": i.get("sub_total_x"), "sub_qib_x": i.get("sub_qib_x"), "sub_nii_x": i.get("sub_nii_x"),
            "sub_retail_x": i.get("sub_retail_x"), "issue_size_cr": i.get("issue_size_cr"),
            "daily_quadrant": None, "daily_since": None, "rs_ratio": None, "rs_momentum": None,
            "first_improving": None, "gain_since_first_improving_pct": None,
            "latest_improving": None, "gain_since_latest_improving_pct": None, "weekly_quadrant": None,
        }
        if s["status"] == "ready":
            daily, _ = stock_rrg(t, "Daily")
            if len(daily):
                q = daily["Quadrant"]
                run_start = q.index[-1]
                for d in reversed(q.index):
                    if q[d] != q.iloc[-1]:
                        break
                    run_start = d
                tu = turns_to_improving(daily)
                row.update({"daily_quadrant": q.iloc[-1], "daily_since": str(run_start.date()),
                            "rs_ratio": round(float(daily["RS_Ratio"].iloc[-1]), 2),
                            "rs_momentum": round(float(daily["RS_Momentum"].iloc[-1]), 2)})
                if len(tu):
                    a, b = tu.index[0], tu.index[-1]
                    row.update({"first_improving": str(a.date()),
                                "gain_since_first_improving_pct": round((last / f.at[a, "Close"] - 1) * 100, 2),
                                "latest_improving": str(b.date()),
                                "gain_since_latest_improving_pct": round((last / f.at[b, "Close"] - 1) * 100, 2)})
            weekly, w_ok = stock_rrg(t, "Weekly")
            if w_ok:
                row["weekly_quadrant"] = weekly["Quadrant"].iloc[-1]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("listing_date", ascending=False)
    df.to_csv(summary_path(), index=False)
    return df


def load_summary() -> pd.DataFrame:
    if not os.path.exists(summary_path()):
        return build_summary()
    return pd.read_csv(summary_path())


if __name__ == "__main__":
    d = build_summary()
    print(len(d), "stocks;", d["daily_quadrant"].value_counts().to_dict())

"""
Test 2: Independent Recomputation of RRG Indicators.
Recomputes RS-Ratio and RS-Momentum using raw pandas operations (WITHOUT
importing the app's calculation functions) and asserts exact agreement
within 1e-6 for all stocks on both Daily and Weekly timeframes.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data_pipeline as dp
import rrg_app as app


def independent_calc(s_stock: pd.Series, s_bench: pd.Series):
    """Pure pandas calculation without importing app functions."""
    rs = s_stock / s_bench
    sm = rs.rolling(5, min_periods=1).mean()
    f = sm.ewm(span=5, adjust=False).mean()
    s = sm.ewm(span=30, adjust=False).mean()
    rr = 100.0 * f / s
    mf = rr.ewm(span=3, adjust=False).mean()
    ms = rr.ewm(span=14, adjust=False).mean()
    rm = 100.0 * mf / ms
    return rr, rm


def test_independent_recomputation_all_stocks():
    bench_df = dp.load_clean_data(cfg.BENCHMARK_TICKER)
    assert bench_df is not None

    for ticker, s_info in cfg.STOCKS.items():
        stock_df = dp.load_clean_data(ticker)
        assert stock_df is not None
        stock_name = s_info["name"]

        aligned = dp.align_stock_with_benchmark(stock_df, bench_df, stock_name, cfg.BENCHMARK_NAME)

        # 1. Daily comparison
        daily_app, d_ok = app.compute_rrg(aligned, stock_name, tf="Daily")
        assert d_ok

        raw_rr, raw_rm = independent_calc(aligned[stock_name], aligned[cfg.BENCHMARK_NAME])
        warmup_d = min(cfg.WARMUP_DAILY, len(aligned) // 3)
        raw_df = pd.DataFrame({"RS_Ratio": raw_rr, "RS_Momentum": raw_rm}).iloc[warmup_d:]

        # Compare last 5 dates
        for dt in daily_app.index[-5:]:
            app_rr = daily_app.loc[dt, "RS_Ratio"]
            app_rm = daily_app.loc[dt, "RS_Momentum"]
            indep_rr = raw_df.loc[dt, "RS_Ratio"]
            indep_rm = raw_df.loc[dt, "RS_Momentum"]
            assert abs(app_rr - indep_rr) < 1e-6, f"Daily RS-Ratio mismatch for {ticker} on {dt}"
            assert abs(app_rm - indep_rm) < 1e-6, f"Daily RS-Momentum mismatch for {ticker} on {dt}"

        # 2. Weekly comparison
        weekly_app, w_ok = app.compute_rrg(aligned, stock_name, tf="Weekly")
        if w_ok:
            wk = aligned.groupby(aligned.index.to_period("W-FRI")).tail(1)
            raw_w_rr, raw_w_rm = independent_calc(wk[stock_name], wk[cfg.BENCHMARK_NAME])
            warmup_w = min(cfg.WARMUP_WEEKLY, len(wk) // 3)
            raw_w_df = pd.DataFrame({"RS_Ratio": raw_w_rr, "RS_Momentum": raw_w_rm}).iloc[warmup_w:]

            for dt in weekly_app.index[-5:]:
                app_rr = weekly_app.loc[dt, "RS_Ratio"]
                app_rm = weekly_app.loc[dt, "RS_Momentum"]
                indep_rr = raw_w_df.loc[dt, "RS_Ratio"]
                indep_rm = raw_w_df.loc[dt, "RS_Momentum"]
                assert abs(app_rr - indep_rr) < 1e-6, f"Weekly RS-Ratio mismatch for {ticker} on {dt}"
                assert abs(app_rm - indep_rm) < 1e-6, f"Weekly RS-Momentum mismatch for {ticker} on {dt}"

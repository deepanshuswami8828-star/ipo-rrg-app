"""
Stocks and tags of the IPO RRG app.
  - tags are the listing year (agreed with the user): every stock has exactly one tag, IPO <listing year>
  - ready = at least MIN_HISTORY_ROWS NSE sessions; waiting = fewer; config.STOCKS holds only ready stocks
  - every stock and NIFTY 50 has a price file; every stock date has a NIFTY 50 close
  - the tag list (config.TAGS) is ALL + exactly the tags of the ready stocks, and each stock appears once
  - prices are positive, OHLC consistent, dates unique and increasing
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
import data_pipeline as dp


def test_tag_is_listing_year():
    for t, s in cfg.ALL_STOCKS.items():
        assert s["tags"] == ["IPO %d" % pd.Timestamp(s["listing_date"]).year], t


def test_ready_and_waiting_follow_sessions():
    for t, s in cfg.ALL_STOCKS.items():
        rows = len(dp.load_clean_data(t))
        assert s["status"] == ("ready" if rows >= cfg.MIN_HISTORY_ROWS else "waiting"), (t, rows, s["status"])
    assert set(cfg.STOCKS) == {t for t, s in cfg.ALL_STOCKS.items() if s["status"] == "ready"}
    assert set(cfg.WAITING) == {t for t, s in cfg.ALL_STOCKS.items() if s["status"] != "ready"}


def test_tag_list_and_each_stock_once():
    tags = {tag for s in cfg.STOCKS.values() for tag in s["tags"]}
    assert cfg.TAGS == [cfg.TAG_ALL] + sorted(tags)
    seen = [t for tag in cfg.TAGS[1:] for t in cfg.stocks_for_tag(tag)]
    assert sorted(seen) == sorted(cfg.STOCKS)               # every ready stock in exactly one tag
    assert cfg.stocks_for_tag(cfg.TAG_ALL) == list(cfg.STOCKS)


def test_price_files_and_nifty_dates():
    nifty = dp.load_clean_data(cfg.BENCHMARK_TICKER)
    assert nifty is not None and len(nifty) >= 100
    for t in cfg.ALL_STOCKS:
        f = dp.load_price_frame(t)
        assert f is not None and len(f) > 0, t
        assert f.index.is_monotonic_increasing and f.index.is_unique, t
        assert (f[["Open", "High", "Low", "Close"]] > 0).all().all(), t
        assert (f["High"] + 1e-6 >= f[["Open", "Close", "Low"]].max(axis=1)).all(), t
        assert (f["Low"] - 1e-6 <= f[["Open", "Close", "High"]].min(axis=1)).all(), t
        missing = f.index.difference(nifty.index)
        assert len(missing) == 0, (t, list(missing[:5]))


def test_manifest_covers_every_stock():
    m = dp.load_manifest()
    for t in list(cfg.ALL_STOCKS) + [cfg.BENCHMARK_TICKER]:
        assert t in m["tickers"], t
        assert m["tickers"][t]["rows"] == len(dp.load_price_frame(t)), t

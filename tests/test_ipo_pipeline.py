"""
IPO RRG data pipeline (NSE official files). Every test runs on a COPY of data/ (IPO_RRG_DATA_DIR), never the real files.

Offline tests (NSE files mocked):
  - a normal session is added to every stock and NIFTY 50
  - NSE re-based previous close (split / bonus): earlier prices scaled by NSE's own factor
  - a >= 40% move without re-basing is HELD (stock file unchanged) and stays held on the next refresh
  - one stock's error never stops the run; the other stocks are updated
  - a waiting stock becomes ready at MIN_HISTORY_ROWS sessions
  - no NSE bhavcopy -> Yahoo fallback rows, marked, then replaced by NSE's figures on a later refresh
Online test (downloads NSE files): drop the last 3 sessions from a copy and refresh: the result must equal the
checked data exactly.
"""
import importlib
import json
import os
import shutil
import sys
from datetime import date, datetime

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
REAL_DATA = os.path.join(ROOT, "data")


def load(tmp_path, monkeypatch, keep=None):
    """Copy data/ to tmp (optionally only some stocks) and reload config + pipeline pointing at it."""
    d = tmp_path / "data"
    shutil.copytree(REAL_DATA, d, ignore=shutil.ignore_patterns("backup", "provisional", ".refresh.lock"))
    if keep:
        s = json.load(open(d / "stocks.json", encoding="utf-8"))
        s = {t: v for t, v in s.items() if t in keep}
        json.dump(s, open(d / "stocks.json", "w", encoding="utf-8"))
    monkeypatch.setenv("IPO_RRG_DATA_DIR", str(d))
    import config as cfg
    import data_pipeline as dp
    importlib.reload(cfg)
    importlib.reload(dp)
    return cfg, dp, d


@pytest.fixture(autouse=True)
def restore_modules():
    yield
    os.environ.pop("IPO_RRG_DATA_DIR", None)
    import config as cfg
    import data_pipeline as dp
    importlib.reload(cfg)
    importlib.reload(dp)


def trim(d, before):
    """Remove every row on or after `before` from the copied price files."""
    for f in list((d / "prices").glob("*.csv")) + [d / "NIFTY50.csv"]:
        x = pd.read_csv(f, index_col=0)
        x[x.index < before].to_csv(f)


def bhav_rows(rows):
    return pd.DataFrame([{"SYMBOL": s, "SERIES": "EQ", "DATE1": dt, "PREV_CLOSE": p, "OPEN_PRICE": o, "HIGH_PRICE": h,
                          "LOW_PRICE": lo, "CLOSE_PRICE": c, "TTL_TRD_QNTY": v} for s, dt, p, o, h, lo, c, v in rows])


NOW = datetime(2026, 11, 3, 22, 0)
DAY = date(2026, 11, 3)
TS = pd.Timestamp(DAY)


def last_raw(d, sym, before="2026-11-03"):
    x = pd.read_csv(d / "prices" / f"{sym}.csv", index_col=0)
    return float(x[x.index < before]["RawClose"].iloc[-1])


def mock_nse(monkeypatch, dp, rows, nifty=True):
    monkeypatch.setattr(dp, "fetch_nse_nifty", lambda d: {"Open": 1, "High": 2, "Low": 1, "Close": 2, "Source": "NSE index file"}
                        if (nifty and d == DAY) else {})
    monkeypatch.setattr(dp, "fetch_nse_bhavcopy", lambda d: bhav_rows(rows) if d == DAY else pd.DataFrame())


def test_normal_session_added(tmp_path, monkeypatch):
    cfg, dp, d = load(tmp_path, monkeypatch, keep=["SWIGGY.NS", "MEESHO.NS"])
    trim(d, "2026-11-03")
    sw, me = last_raw(d, "SWIGGY"), last_raw(d, "MEESHO")
    mock_nse(monkeypatch, dp, [("SWIGGY", "03-Nov-2026", sw, sw, sw * 1.02, sw * 0.99, sw * 1.01, 1000),
                               ("MEESHO", "03-Nov-2026", me, me, me, me, me * 0.98, 2000)])
    r = dp.refresh_data(force=True, now_ist=NOW.replace(tzinfo=cfg.IST))
    assert r["status"] == "SUCCESS", r["message"]
    for s, prev, mult in (("SWIGGY", sw, 1.01), ("MEESHO", me, 0.98)):
        f = dp.load_price_frame(s + ".NS")
        assert f.index[-1] == TS and f["Source"].iloc[-1] == "NSE bhavcopy"
        assert abs(f["Close"].iloc[-1] - prev * mult) < 1e-9
        assert len(f) == len(f.index.unique())
    assert dp.load_price_frame("^NSEI").index[-1] == TS


def test_split_rebased_by_nse(tmp_path, monkeypatch):
    cfg, dp, d = load(tmp_path, monkeypatch, keep=["SWIGGY.NS"])
    trim(d, "2026-11-03")
    before = dp.load_price_frame("SWIGGY.NS")
    raw = last_raw(d, "SWIGGY")
    mock_nse(monkeypatch, dp, [("SWIGGY", "03-Nov-2026", raw / 2, raw / 2, raw / 2, raw / 2, raw / 2 * 1.01, 500)])
    r = dp.refresh_data(force=True, now_ist=NOW.replace(tzinfo=cfg.IST))
    after = dp.load_price_frame("SWIGGY.NS")
    assert "SWIGGY.NS" in r["report"]["rebased"]
    old = after[after.index < TS]
    assert (abs(old["Close"] - before["Close"] / 2) < 1e-6).all()           # history halved, as NSE did
    assert abs(after["Close"].iloc[-1] / old["Close"].iloc[-1] - 1.01) < 1e-9  # the day itself is an ordinary +1%


def test_big_move_without_rebase_is_held(tmp_path, monkeypatch):
    cfg, dp, d = load(tmp_path, monkeypatch, keep=["SWIGGY.NS", "MEESHO.NS"])
    trim(d, "2026-11-03")
    sw_file = (d / "prices" / "SWIGGY.csv").read_bytes()
    sw, me = last_raw(d, "SWIGGY"), last_raw(d, "MEESHO")
    mock_nse(monkeypatch, dp, [("SWIGGY", "03-Nov-2026", sw, sw / 10, sw / 10, sw / 10, sw / 10, 9999),   # -90%, no re-base
                               ("MEESHO", "03-Nov-2026", me, me, me, me, me, 10)])
    r = dp.refresh_data(force=True, now_ist=NOW.replace(tzinfo=cfg.IST))
    assert r["status"] == "SUCCESS"
    assert "SWIGGY.NS" in r["report"]["held"]
    assert (d / "prices" / "SWIGGY.csv").read_bytes() == sw_file             # held stock untouched
    assert dp.load_price_frame("MEESHO.NS").index[-1] == TS                 # the other stock still updated
    assert dp.load_manifest()["tickers"]["SWIGGY.NS"]["status"] == "HELD"
    # next refresh: still held, nothing added to it
    nxt = date(2026, 11, 4)
    monkeypatch.setattr(dp, "fetch_nse_nifty", lambda x: {"Open": 1, "High": 2, "Low": 1, "Close": 2, "Source": "NSE index file"} if x == nxt else {})
    monkeypatch.setattr(dp, "fetch_nse_bhavcopy", lambda x: bhav_rows([("SWIGGY", "04-Nov-2026", sw / 10, 1, 1, 1, 1, 1)]) if x == nxt else pd.DataFrame())
    r2 = dp.refresh_data(force=True, now_ist=datetime(2026, 11, 4, 22, 0, tzinfo=cfg.IST))
    assert r2["report"]["held"]["SWIGGY.NS"].startswith("still held")
    assert (d / "prices" / "SWIGGY.csv").read_bytes() == sw_file


def test_one_stock_error_does_not_stop_the_run(tmp_path, monkeypatch):
    cfg, dp, d = load(tmp_path, monkeypatch, keep=["SWIGGY.NS", "MEESHO.NS"])
    trim(d, "2026-11-03")
    sw, me = last_raw(d, "SWIGGY"), last_raw(d, "MEESHO")
    mock_nse(monkeypatch, dp, [("SWIGGY", "03-Nov-2026", sw, sw, sw, sw, sw, 1),
                               ("MEESHO", "03-Nov-2026", me, me, me, me, me, 1)])
    real = dp._stock_update

    def broken(info, frame, ts, row):
        if info["symbol"] == "SWIGGY":
            raise RuntimeError("simulated failure")
        return real(info, frame, ts, row)
    monkeypatch.setattr(dp, "_stock_update", broken)
    r = dp.refresh_data(force=True, now_ist=NOW.replace(tzinfo=cfg.IST))
    assert r["status"] == "SUCCESS"
    assert "SWIGGY.NS" in r["report"]["errors"]
    assert dp.load_price_frame("MEESHO.NS").index[-1] == TS


def test_waiting_stock_promoted_at_60_sessions(tmp_path, monkeypatch):
    cfg, dp, d = load(tmp_path, monkeypatch)
    s = json.load(open(d / "stocks.json", encoding="utf-8"))
    waiting = [t for t, v in s.items() if v["status"] == "waiting"]
    t = waiting[0]
    sym = s[t]["symbol"]
    json.dump({t: s[t]}, open(d / "stocks.json", "w", encoding="utf-8"))
    f = pd.read_csv(d / "prices" / f"{sym}.csv", index_col=0)
    f = f[f.index < "2026-11-03"]
    # make it 59 sessions by repeating its first row on earlier weekdays (test data only)
    pad = pd.bdate_range(end=pd.Timestamp(f.index[0]) - pd.Timedelta(days=1), periods=59 - len(f))
    f = pd.concat([pd.DataFrame([f.iloc[0].to_dict()] * len(pad), index=pad.strftime("%Y-%m-%d")), f])
    f.index.name = "Date"
    f.to_csv(d / "prices" / f"{sym}.csv")
    n = pd.read_csv(d / "NIFTY50.csv", index_col=0)
    n[n.index < "2026-11-03"].to_csv(d / "NIFTY50.csv")
    raw = float(f["RawClose"].iloc[-1])
    mock_nse(monkeypatch, dp, [(sym, "03-Nov-2026", raw, raw, raw, raw, raw, 1)])
    r = dp.refresh_data(force=True, now_ist=NOW.replace(tzinfo=cfg.IST))
    assert t in r["report"]["promoted"]
    assert json.load(open(d / "stocks.json", encoding="utf-8"))[t]["status"] == "ready"


def test_yahoo_fallback_then_replaced_by_nse(tmp_path, monkeypatch):
    cfg, dp, d = load(tmp_path, monkeypatch, keep=["SWIGGY.NS"])
    trim(d, "2026-11-03")
    sw = last_raw(d, "SWIGGY")
    # day 1: index file out, bhavcopy not -> Yahoo fallback row
    monkeypatch.setattr(dp, "fetch_nse_nifty", lambda x: {"Open": 1, "High": 2, "Low": 1, "Close": 2, "Source": "NSE index file"} if x == DAY else {})
    monkeypatch.setattr(dp, "fetch_nse_bhavcopy", lambda x: pd.DataFrame())
    monkeypatch.setattr(dp, "yahoo_row", lambda t, x: {"Open": sw, "High": sw, "Low": sw, "Close": sw * 1.5, "Volume": 1,
                                                        "RawClose": sw * 1.5, "Source": dp.FALLBACK})
    r = dp.refresh_data(force=True, now_ist=NOW.replace(tzinfo=cfg.IST))
    f = dp.load_price_frame("SWIGGY.NS")
    assert r["report"]["fallback"] and f["Source"].iloc[-1] == dp.FALLBACK
    assert dp.load_manifest()["tickers"]["SWIGGY.NS"]["yahoo_fallback_days"] == ["2026-11-03"]
    # day 2: NSE's file for day 1 is out -> the fallback row is replaced by NSE's figure
    mock_nse(monkeypatch, dp, [("SWIGGY", "03-Nov-2026", sw, sw, sw, sw, sw * 1.01, 7)])
    r2 = dp.refresh_data(force=True, now_ist=datetime(2026, 11, 4, 9, 0, tzinfo=cfg.IST))
    f2 = dp.load_price_frame("SWIGGY.NS")
    assert f2.at[TS, "Source"] == "NSE bhavcopy" and abs(f2.at[TS, "Close"] - sw * 1.01) < 1e-9
    assert dp.load_manifest()["tickers"]["SWIGGY.NS"]["yahoo_fallback_days"] == []


@pytest.mark.online
def test_round_trip_with_real_nse_files(tmp_path, monkeypatch):
    """Downloads NSE's files: dropping the last 3 sessions and refreshing must give back the checked data exactly."""
    cfg, dp, d = load(tmp_path, monkeypatch)
    orig = {f.name: pd.read_csv(f, index_col=0) for f in list((d / "prices").glob("*.csv")) + [d / "NIFTY50.csv"]}
    last = sorted(pd.read_csv(d / "NIFTY50.csv", index_col=0).index)[-3]
    trim(d, last)
    r = dp.refresh_data(force=True, now_ist=datetime.now(cfg.IST))
    assert r["status"] == "SUCCESS", r["message"]
    for name, o in orig.items():
        p = d / ("prices" if name != "NIFTY50.csv" else "") / name
        n = pd.read_csv(p, index_col=0)
        assert list(n.index) == list(o.index), name
        assert (n[["Open", "High", "Low", "Close"]] - o[["Open", "High", "Low", "Close"]]).abs().max().max() < 1e-6, name

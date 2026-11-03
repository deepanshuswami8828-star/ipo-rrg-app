"""
Data pipeline for the IPO RRG app. Zero Streamlit imports.

Prices come from NSE's official archive files:
  - stocks: the daily bhavcopy sec_bhavdata_full_DDMMYYYY.csv (series EQ / BE), split-adjusted to today's shares
  - NIFTY 50: the daily index file ind_close_all_DDMMYYYY.csv
Yahoo Finance is used only as a fallback for a session whose NSE file is not available, and every such row is
marked (Source column, manifest) and replaced by NSE's figure on a later refresh once the file is out.

One stock's problem never stops the refresh: that stock keeps its stored data and the manifest says why
(rule of the project: the run covers every stock, failures are reported, never silent).

Storage (data/):
  prices/<SYMBOL>.csv  Date, Open, High, Low, Close, Volume, RawClose, Source   (prices split-adjusted)
  NIFTY50.csv          Date, Open, High, Low, Close, Source
  stocks.json          stocks and their status (ready / waiting); a waiting stock becomes ready at MIN_HISTORY_ROWS
  manifest.json        version, last refresh, per-ticker coverage, warnings, held stocks, fallbacks
"""
import hashlib
import importlib
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import tempfile
import time as pytime
import urllib.error
import urllib.request
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
from config import (
    DATA_DIR, PRICES_DIR, LOGS_DIR, MANIFEST_PATH, STOCKS_JSON, REFRESH_LOG_PATH, LOCK_FILE_PATH, IST,
    BENCHMARK_TICKER, BENCHMARK_NAME, MARKET_OPEN_IST, MARKET_CLOSE_IST, REFRESH_AFTER_CLOSE_IST,
    MIN_REFRESH_INTERVAL_MINUTES, SUSPICIOUS_MOVE_PCT, HOLD_MOVE_PCT, MIN_HISTORY_ROWS, PROVIDER,
    NSE_BHAV_URL, NSE_INDEX_URL, NSE_SERIES, get_clean_csv_path, sanitize_ticker
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PRICES_DIR, exist_ok=True)

logger = logging.getLogger("ipo_rrg_data_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    rf_handler = RotatingFileHandler(REFRESH_LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    rf_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(rf_handler)

PRICE_COLS = ["Open", "High", "Low", "Close"]
FALLBACK = "Yahoo fallback"
RECHECK_DAYS = 10        # fallback rows younger than this are re-tried from NSE on every refresh


class FileLock:
    """File lock to prevent concurrent refresh processes."""
    def __init__(self, lock_path: str = LOCK_FILE_PATH, timeout_sec: float = 15.0):
        self.lock_path = lock_path
        self.timeout_sec = timeout_sec
        self.acquired = False

    def __enter__(self):
        start_time = pytime.time()
        while pytime.time() - start_time < self.timeout_sec:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode("utf-8"))
                self.acquired = True
                return self
            except FileExistsError:
                pytime.sleep(0.2)
        try:                                     # stale lock (> 10 minutes old)
            if pytime.time() - os.path.getmtime(self.lock_path) > 600:
                logger.warning("Breaking stale lock file %s", self.lock_path)
                os.remove(self.lock_path)
                return self.__enter__()
        except Exception:
            pass
        raise TimeoutError(f"Could not acquire file lock on {self.lock_path}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            try:
                os.close(self.fd)
                if os.path.exists(self.lock_path):
                    os.remove(self.lock_path)
            except Exception as e:
                logger.error("Error releasing lock: %s", e)
            self.acquired = False


def calculate_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---- manifest -------------------------------------------------------------------------------------
def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to parse manifest: %s", e)
    return {"version_id": "none", "updated_at": None, "status": "INITIAL", "provider": PROVIDER, "tickers": {},
            "last_attempt": {"time": None, "result": "NONE", "message": "No refresh attempt recorded yet"}}


def _atomic_write_text(path: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(path), delete=False, encoding="utf-8", newline="") as tf:
        tf.write(text)
        tmp = tf.name
    os.replace(tmp, path)


def save_manifest(manifest: dict) -> None:
    _atomic_write_text(MANIFEST_PATH, json.dumps(manifest, indent=2, default=str))


def _write_frame(df: pd.DataFrame, path: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index_label="Date", date_format="%Y-%m-%d")
    _atomic_write_text(path, buf.getvalue())


# ---- loading ----------------------------------------------------------------------------------------
def load_price_frame(ticker: str) -> Optional[pd.DataFrame]:
    """Full stored frame of a ticker (all columns), or None."""
    path = get_clean_csv_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.sort_index()
    except Exception as e:
        logger.error("Error loading %s: %s", path, e)
        return None


def load_clean_data(ticker: str) -> Optional[pd.DataFrame]:
    """DataFrame with a DatetimeIndex and a Close column (split-adjusted), or None."""
    df = load_price_frame(ticker)
    if df is None or "Close" not in df.columns:
        return None
    return df[["Close"]]


def _load_column(ticker: str, column: str) -> Optional[pd.Series]:
    df = load_price_frame(ticker)
    if df is None or column not in df.columns:
        return None
    return df[column].dropna()


def load_high_data(ticker: str) -> Optional[pd.Series]:
    """Daily high (split-adjusted like the close)."""
    return _load_column(ticker, "High")


def load_open_data(ticker: str) -> Optional[pd.Series]:
    """Daily open (split-adjusted like the close)."""
    return _load_column(ticker, "Open")


def refresh_open_high(tickers=None, **_) -> Dict[str, str]:
    """Kept for the RRG app's interface: open and high are stored with the close, nothing extra to download."""
    return {t: "OK (stored with the close)" for t in (tickers or list(cfg.STOCKS))}


def align_stock_with_benchmark(stock_df: pd.DataFrame, bench_df: pd.DataFrame, stock_name: str,
                               bench_name: str = BENCHMARK_NAME) -> pd.DataFrame:
    """Strict INNER JOIN between stock and benchmark on Date. No forward fill, no back fill, no interpolation."""
    aligned = pd.concat([stock_df["Close"].rename(stock_name), bench_df["Close"].rename(bench_name)],
                        axis=1, join="inner").dropna()
    return aligned.sort_index()


# ---- market time ------------------------------------------------------------------------------------
def is_market_open(now_dt: datetime) -> bool:
    return now_dt.weekday() < 5 and (MARKET_OPEN_IST <= now_dt.time() <= MARKET_CLOSE_IST)


def is_session_final(candle_date: date, now_dt: datetime) -> bool:
    if candle_date < now_dt.date():
        return True
    return candle_date == now_dt.date() and now_dt.time() >= REFRESH_AFTER_CLOSE_IST


def get_expected_latest_session(now_dt: datetime) -> date:
    """Latest weekday whose session should be complete (NSE holidays are not known here)."""
    if now_dt.weekday() < 5 and now_dt.time() >= REFRESH_AFTER_CLOSE_IST:
        return now_dt.date()
    d = now_dt.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ---- NSE files -----------------------------------------------------------------------------------------
def _download(url: str, retries: int = 3) -> Optional[bytes]:
    """File body; b'' when NSE has no file (404); None when the download kept failing."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
        except Exception as e:
            logger.warning("download %s failed (attempt %d): %s", url, attempt + 1, e)
        pytime.sleep(2 * (attempt + 1))
    return None


def fetch_nse_bhavcopy(d: date) -> Optional[pd.DataFrame]:
    """NSE detailed bhavcopy of a day (EQ/BE rows), empty frame if NSE has no file, None on download failure.
    A file whose own date differs from d (NSE sometimes serves an older day on holidays) counts as no file."""
    body = _download(NSE_BHAV_URL % d.strftime("%d%m%Y"))
    if body is None:
        return None
    if not body.strip() or body[:2] == b"PK":
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(body), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for c in ("SYMBOL", "SERIES", "DATE1"):
        df[c] = df[c].astype(str).str.strip()
    if len(df) and pd.to_datetime(df["DATE1"].iloc[0], format="%d-%b-%Y").date() != d:
        return pd.DataFrame()
    return df[df["SERIES"].isin(NSE_SERIES)]


def fetch_nse_nifty(d: date) -> Optional[dict]:
    """NIFTY 50 row of NSE's index file for day d; {} if NSE has no file; None on download failure."""
    body = _download(NSE_INDEX_URL % d.strftime("%d%m%Y"))
    if body is None:
        return None
    if not body.strip():
        return {}
    t = pd.read_csv(io.BytesIO(body))
    t.columns = [c.strip() for c in t.columns]
    n = t[t["Index Name"].astype(str).str.strip().str.lower() == "nifty 50"]
    if not len(n) or pd.to_datetime(n.iloc[0]["Index Date"], format="%d-%m-%Y").date() != d:
        return {}
    x = n.iloc[0]
    return {"Open": float(x["Open Index Value"]), "High": float(x["High Index Value"]),
            "Low": float(x["Low Index Value"]), "Close": float(x["Closing Index Value"]), "Source": "NSE index file"}


def yahoo_row(ticker: str, d: date) -> Optional[dict]:
    """Fallback: one day's Yahoo prices (split-adjusted, not dividend-adjusted), or None."""
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(start=d, end=d + timedelta(days=1), auto_adjust=False, timeout=30)
        if len(h):
            r = h.iloc[-1]
            return {"Open": float(r["Open"]), "High": float(r["High"]), "Low": float(r["Low"]),
                    "Close": float(r["Close"]), "Volume": int(r["Volume"]), "RawClose": float(r["Close"]), "Source": FALLBACK}
    except Exception as e:
        logger.warning("Yahoo fallback %s %s failed: %s", ticker, d, e)
    return None


# ---- refresh ------------------------------------------------------------------------------------------------
def _stock_update(info: dict, frame: pd.DataFrame, d: pd.Timestamp, row: Optional[pd.Series]) -> Tuple[pd.DataFrame, str]:
    """Add NSE's row for day d to a stock's frame. Returns (frame, note); note '' when all is normal.
    - NSE re-based the previous close (split / bonus): earlier prices are scaled by NSE's own factor.
    - a move of HOLD_MOVE_PCT or more without such re-basing is held (not added) and reported for review."""
    if row is None:
        return frame, "no NSE row (no trade or suspended)"
    prev, close = float(row["PREV_CLOSE"]), float(row["CLOSE_PRICE"])
    note = ""
    last = frame[frame.index < d]
    if len(last):
        last_raw = float(last["RawClose"].iloc[-1])
        if abs(prev / last_raw - 1) > 0.005:
            f = prev / last_raw
            earlier = frame.index < d
            frame.loc[earlier, PRICE_COLS] = frame.loc[earlier, PRICE_COLS] * f
            frame.loc[earlier, "Volume"] = (frame.loc[earlier, "Volume"] / f).round(0)
            note = "NSE re-based the previous close %.2f -> %.2f (x%.4f, split/bonus/other action): earlier prices adjusted" % (last_raw, prev, f)
        elif abs(close / prev - 1) * 100 >= HOLD_MOVE_PCT:
            return frame, "HELD: %+.1f%% move on %s without NSE re-basing the previous close (possible split/bonus) - needs review" % (
                (close / prev - 1) * 100, d.date())
    frame.loc[d] = {"Open": float(row["OPEN_PRICE"]), "High": float(row["HIGH_PRICE"]), "Low": float(row["LOW_PRICE"]),
                    "Close": close, "Volume": int(float(row["TTL_TRD_QNTY"])), "RawClose": close, "Source": "NSE bhavcopy"}
    return frame.sort_index(), note


def refresh_data(force: bool = False, include_provisional: bool = False, now_ist: Optional[datetime] = None,
                 max_days: int = 15) -> dict:
    """Add every completed NSE session after the stored data (and re-try recent Yahoo fallback rows from NSE).
    Returns {"status": SUCCESS | NO_NEW_SESSION | SKIPPED | LOCKED | FAILED, "message", "manifest", "report"}."""
    now_ist = now_ist or datetime.now(IST)
    manifest = load_manifest()
    last_fetched = manifest.get("updated_at")
    if not force and last_fetched:
        try:
            if (now_ist - datetime.fromisoformat(last_fetched)).total_seconds() < MIN_REFRESH_INTERVAL_MINUTES * 60:
                return {"status": "SKIPPED", "message": "last refresh under %d minutes ago" % MIN_REFRESH_INTERVAL_MINUTES,
                        "manifest": manifest, "report": {}}
        except Exception:
            pass

    def attempt(result, msg, report):
        manifest["last_attempt"] = {"time": now_ist.isoformat(), "result": result, "message": msg}
        save_manifest(manifest)
        logger.info("%s: %s", result, msg)
        return {"status": result, "message": msg, "manifest": manifest, "report": report}

    try:
        with FileLock():
            stocks = cfg.load_all_stocks()
            nifty = load_price_frame(BENCHMARK_TICKER)
            if nifty is None or not len(nifty):
                return attempt("FAILED", "no stored NIFTY 50 data (run tools/build_app_data.py first)", {})
            frames = {t: load_price_frame(t) for t in stocks}
            report = {"sessions_added": [], "fallback": [], "held": {}, "rebased": {}, "no_row": {}, "errors": {},
                      "promoted": [], "missing_files": []}
            # a stock held by an earlier refresh stays held (no new rows) until someone reviews it and clears
            # "held" in manifest.json; adding later rows would misread the unreviewed price jump
            for t, v in manifest.get("tickers", {}).items():
                if v.get("held") and t in stocks:
                    report["held"][t] = "still held: " + v["held"].replace("still held: ", "")

            # days to fetch: after the stored NIFTY date, plus recent days still holding Yahoo fallback rows
            last_day = nifty.index[-1].date()
            days = []
            d = last_day + timedelta(days=1)
            while d <= now_ist.date() and len(days) < max_days:
                if is_session_final(d, now_ist):
                    days.append(d)
                d += timedelta(days=1)
            redo = sorted({x.date() for f in [nifty, *frames.values()] if f is not None and "Source" in f
                           for x in f.index[(f["Source"] == FALLBACK) & (f.index >= pd.Timestamp(now_ist.date()) - pd.Timedelta(days=RECHECK_DAYS))]})
            if not days and not redo:
                return attempt("NO_NEW_SESSION", "data already up to %s; no completed session after it" % last_day, report)

            changed = False
            for d in sorted(set(days) | set(redo)):
                ts = pd.Timestamp(d)
                idx = fetch_nse_nifty(d)
                bhav = fetch_nse_bhavcopy(d)
                is_redo = d in redo and d not in days
                if is_redo and (not idx or bhav is None or not len(bhav)):
                    continue                                   # NSE still has no file: keep the fallback rows
                if not idx and (bhav is None or not len(bhav)):
                    # no NSE file at all: a holiday, or not published yet (stop at the first such recent day)
                    if (now_ist.date() - d).days <= 3 and d.weekday() < 5:
                        report["missing_files"].append(str(d))
                        break
                    continue
                if not idx:                                    # stock file out, index file not: NIFTY from Yahoo
                    idx = yahoo_row(BENCHMARK_TICKER, d)
                    if idx is None:
                        report["errors"][BENCHMARK_TICKER] = "no NSE index file and no Yahoo price for %s" % d
                        break
                    idx = {k: idx[k] for k in PRICE_COLS + ["Source"]}
                    report["fallback"].append("%s NIFTY 50" % d)
                nifty.loc[ts] = idx
                nifty = nifty.sort_index()
                use_yahoo = bhav is None or not len(bhav)
                if use_yahoo:
                    report["fallback"].append("%s all stocks (no NSE bhavcopy)" % d)
                by_sym = {} if use_yahoo else {s: r for s, r in zip(bhav["SYMBOL"], bhav.to_dict("records"))}
                for t, info in stocks.items():
                    if t in report["held"]:
                        continue
                    try:
                        f = frames.get(t)
                        if f is None:
                            report["errors"][t] = "no stored data"
                            continue
                        if use_yahoo:
                            y = yahoo_row(t, d)
                            if y is None:
                                report["no_row"].setdefault(t, []).append(str(d))
                                continue
                            f.loc[ts] = y
                            frames[t] = f.sort_index()
                            continue
                        row = by_sym.get(info["symbol"])
                        if is_redo and ts in f.index:
                            f = f.drop(ts)                     # replace the Yahoo fallback row with NSE's
                        f, note = _stock_update(info, f, ts, pd.Series(row) if row else None)
                        frames[t] = f
                        if note.startswith("HELD"):
                            report["held"][t] = note
                        elif note.startswith("NSE re-based"):
                            report["rebased"][t] = "%s: %s" % (d, note)
                        elif note:
                            report["no_row"].setdefault(t, []).append(str(d))
                    except Exception as e:                     # one stock never stops the run
                        report["errors"][t] = "%s: %s" % (d, e)
                report["sessions_added"].append(str(d) + (" (NSE replaced Yahoo fallback)" if is_redo else ""))
                changed = True

            if not changed:
                msg = "no new NSE session yet" + (" (files missing for %s)" % report["missing_files"] if report["missing_files"] else "")
                return attempt("NO_NEW_SESSION", msg, report)

            # write: a held stock keeps its stored file (nothing after the held day is added to it)
            _write_frame(nifty, get_clean_csv_path(BENCHMARK_TICKER))
            for t, f in frames.items():
                if f is not None and t not in report["held"]:
                    _write_frame(f, get_clean_csv_path(t))
            for t, info in stocks.items():
                f = frames.get(t)
                if f is not None:
                    info["sessions"] = int(len(f))
                    info["last_date"] = str(f.index[-1].date())
                    if info.get("status") != "ready" and len(f) >= MIN_HISTORY_ROWS:
                        info["status"] = "ready"
                        report["promoted"].append(t)
            _atomic_write_text(STOCKS_JSON, json.dumps(stocks, indent=1))
            rebuild_manifest(manifest, now_ist, report)
            try:                                   # per-stock summary the app reads (RRG now, signals, gains)
                import analytics
                importlib.reload(cfg)
                analytics.build_summary()
            except Exception as e:
                report["errors"]["summary"] = "summary not rebuilt: %s" % e
            msg = "added %s; %d stocks; fallback %d; held %d; errors %d; promoted %d" % (
                ", ".join(report["sessions_added"]), len(stocks), len(report["fallback"]), len(report["held"]),
                len(report["errors"]), len(report["promoted"]))
            return attempt("SUCCESS", msg, report)
    except TimeoutError as te:
        return {"status": "LOCKED", "message": "refresh skipped: lock busy (%s)" % te, "manifest": manifest, "report": {}}
    except Exception as exc:
        logger.exception("refresh failed")
        return attempt("FAILED", "unexpected error in refresh: %s" % exc, {})


def rebuild_manifest(manifest: dict, now_ist: datetime, report: Optional[dict] = None) -> dict:
    """Per-ticker coverage, flagged moves and sources from the stored files; a new version id."""
    stocks = cfg.load_all_stocks()
    tickers = {}
    for t in list(stocks) + [BENCHMARK_TICKER]:
        f = load_price_frame(t)
        if f is None or not len(f):
            tickers[t] = {"status": "NO_DATA", "rows": 0, "flagged_moves": []}
            continue
        ret = f["Close"].pct_change() * 100
        flags = [{"date": d.strftime("%Y-%m-%d"), "pct_change": round(float(v), 2), "close": round(float(f.at[d, "Close"]), 2)}
                 for d, v in ret[ret.abs() >= SUSPICIOUS_MOVE_PCT].items()]
        fb = f.index[f["Source"] == FALLBACK] if "Source" in f else []
        held = (report or {}).get("held", {}).get(t) or manifest.get("tickers", {}).get(t, {}).get("held")
        tickers[t] = {"first_date": f.index[0].strftime("%Y-%m-%d"), "last_date": f.index[-1].strftime("%Y-%m-%d"),
                      "rows": int(len(f)), "sha256": calculate_sha256(get_clean_csv_path(t)),
                      "status": "HELD" if held else ("VALID" if t == BENCHMARK_TICKER or stocks[t].get("status") == "ready" else "WAITING"),
                      "held": held or None, "yahoo_fallback_days": [d.strftime("%Y-%m-%d") for d in fb],
                      "flagged_moves": flags, "warnings": []}
    manifest.update({"version_id": now_ist.strftime("v%Y%m%d_%H%M%S"), "updated_at": now_ist.isoformat(), "status": "SUCCESS",
                     "provider": PROVIDER, "tickers": tickers})
    if report is not None:
        manifest["last_report"] = report
    save_manifest(manifest)
    return manifest

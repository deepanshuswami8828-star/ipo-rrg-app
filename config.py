"""
Configuration for the IPO RRG app (NSE mainboard IPOs listed from Nov 2024, against NIFTY 50).

Stocks are not listed here by hand: data/stocks.json holds every IPO stock that passed the 3-condition data
check (built by tools/build_app_data.py in the project folder, kept up to date by the daily refresh).
A stock with status "ready" (at least MIN_HISTORY_ROWS NSE sessions) is on the RRG pages; "waiting" stocks are
shown on the IPO page only, and the refresh moves them to "ready" once they have enough sessions.
"""
import json
import os
from datetime import time, timedelta, timezone

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# IPO_RRG_DATA_DIR lets tests run the pipeline on a copy of the data, never on the real files
DATA_DIR = os.environ.get("IPO_RRG_DATA_DIR") or os.path.join(ROOT_DIR, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")
BACKUP_DIR = os.path.join(DATA_DIR, "backup")
PROVISIONAL_DIR = os.path.join(DATA_DIR, "provisional")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
STOCKS_JSON = os.path.join(DATA_DIR, "stocks.json")
NIFTY_CSV = os.path.join(DATA_DIR, "NIFTY50.csv")
IPO_INFO_CSV = os.path.join(DATA_DIR, "ipo_info.csv")
GMP_CSV = os.path.join(DATA_DIR, "gmp_latest.csv")
REFRESH_LOG_PATH = os.path.join(LOGS_DIR, "refresh.log")
LOCK_FILE_PATH = os.path.join(DATA_DIR, ".refresh.lock")

IST = timezone(timedelta(hours=5, minutes=30))

BENCHMARK_TICKER = "^NSEI"
BENCHMARK_NAME = "NIFTY 50"
PROVIDER = "NSE official (daily bhavcopy + index close file); Yahoo Finance only as a marked fallback"

# NSE archive files used by the refresh
NSE_BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_%s.csv"
NSE_INDEX_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_%s.csv"
NSE_SERIES = ("EQ", "BE")       # mainboard equity series (BE = trade-for-trade)


def load_all_stocks() -> dict:
    """Every IPO stock in data/stocks.json (ready and waiting), keyed by 'SYMBOL.NS'."""
    try:
        with open(STOCKS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


ALL_STOCKS = load_all_stocks()
# stocks on the RRG pages: ready ones, oldest listing first
STOCKS = {t: v for t, v in sorted(ALL_STOCKS.items(), key=lambda kv: (kv[1]["listing_date"], kv[0]))
          if v.get("status") == "ready"}
WAITING = {t: v for t, v in sorted(ALL_STOCKS.items(), key=lambda kv: (kv[1]["listing_date"], kv[0]))
           if v.get("status") != "ready"}

# Tags (agreed with the user, 3 Nov 2026): listing year. "ALL" = every ready stock.
TAG_ALL = "ALL"
TAGS = [TAG_ALL] + sorted({tag for v in STOCKS.values() for tag in v.get("tags", [])})


def stocks_for_tag(tag: str) -> list:
    """Tickers shown for a tag, in the order of STOCKS."""
    if tag == TAG_ALL:
        return list(STOCKS)
    return [t for t, info in STOCKS.items() if tag in info.get("tags", [])]


DATA_START_DISPLAY = "2024-11-01"
MARKET_OPEN_IST = time(9, 15)
MARKET_CLOSE_IST = time(15, 30)
REFRESH_AFTER_CLOSE_IST = time(15, 45)

MIN_REFRESH_INTERVAL_MINUTES = 15
MAX_POST_CLOSE_ATTEMPTS_PER_DAY = 4
POST_CLOSE_RETRY_INTERVAL_MINUTES = 30
SUSPICIOUS_MOVE_PCT = 25.0
HOLD_MOVE_PCT = 40.0     # a one-day move this big without NSE re-basing the previous close is held for review
MIN_HISTORY_ROWS = 60    # NSE sessions a stock needs before it goes on the RRG pages
MIN_WEEKLY_BARS = 60     # weekly RRG bars needed before the Weekly view is shown (Daily is the main view)

# RRG Calculation Parameters (calibrated method, unchanged from the RRG app)
RS_PRESMOOTH = 5     # SMA window on raw RS before the double EMA
RS_SHORT = 5         # fast EMA for RS-Ratio
RS_LONG = 30         # slow EMA for RS-Ratio (baseline)
MOM_SHORT = 3        # fast EMA for RS-Momentum
MOM_LONG = 14        # slow EMA for RS-Momentum
TAIL_DAYS_DAILY = 25 # trading days of trajectory drawn
TAIL_WEEKS = 12      # weekly trail length
WARMUP_DAILY = 200   # daily warm-up bars (capped at a third of the history for young stocks)
WARMUP_WEEKLY = 100  # weekly warm-up bars


def sanitize_ticker(ticker: str) -> str:
    """Removes ^ and the .NS suffix for file naming."""
    return ticker.replace("^", "").replace(".NS", "")


def get_clean_csv_path(ticker: str) -> str:
    """Price file of a ticker: data/NIFTY50.csv for the benchmark, data/prices/<SYMBOL>.csv for a stock."""
    if ticker == BENCHMARK_TICKER:
        return NIFTY_CSV
    return os.path.join(PRICES_DIR, f"{sanitize_ticker(ticker)}.csv")

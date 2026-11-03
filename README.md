# IPO RRG app

Relative Rotation Graphs (RRG) of NSE mainboard IPO stocks listed from November 2024, against NIFTY 50, plus an
IPO details page (listing performance, subscription, GMP of upcoming IPOs, and the "Daily RRG turned Improving after
listing" signal). Built from the RRG app (same RRG method); this folder is the app's repository.

## Run
```
pip install -r requirements.txt
streamlit run streamlit_app.py        # rrg_app.py and ipo_page.py start the same app
```

## Which stocks
Every NSE mainboard IPO listed from 1 Nov 2024 (no SME, no InvIT/REIT) that passed the 3-condition data check
(price break, repricing, missing data) on NSE's official daily files from its listing day. The list and the checks
live in the project folder (`IPO RRG app\Stock data checks`, `Reports\`); `data/stocks.json` is built from them.
- **ready**: at least 60 NSE sessions, shown on the RRG pages
- **waiting**: fewer sessions, shown on the IPO details page only; the daily refresh moves it to ready at 60
- **tags**: the listing year (IPO 2024 / IPO 2025 / IPO 2026)

## Tabs
- **Home**: every IPO in one table (search, tag, Daily RRG filter); click a row to open it in Stock.
- **Stock**: one IPO: issue price -> listing day -> now, Daily RRG chart against NIFTY 50, price since listing with
  the issue price and the days the Daily RRG turned Improving, and every IPO field with its source and read date.
- **Compare**: every IPO of a tag on one Daily RRG chart, with a table.
- **Upcoming IPOs**: GMP of mainboard IPOs not listed yet (UNOFFICIAL, grey market; two sites side by side with the
  time read), and for listed ones the GMP estimate against the actual NSE listing open.
- Daily RRG is the main view for IPO stocks; the Weekly quadrant is shown once a stock has enough history.
- The strategy pages of the RRG app are left out until the strategy is changed.

Code: `app_main.py` (tabs), `analytics.py` (per-stock summary, `data/summary.csv`, rebuilt by every refresh),
`rrg_calc.py` (RRG formula of the RRG app, unchanged), `data_pipeline.py` (NSE data), `config.py`.

## Data (all in data/)
| File | What | Source |
|---|---|---|
| `prices/<SYMBOL>.csv` | Date, Open, High, Low, Close, Volume (split-adjusted), RawClose, Source | NSE daily bhavcopy (`sec_bhavdata_full`) |
| `NIFTY50.csv` | NIFTY 50 daily OHLC | NSE index close file (`ind_close_all`) |
| `stocks.json` | stocks, tag, listing date, issue price, ready/waiting | project step 1 + 2 |
| `ipo_info.csv` | IPO details, each field with source + read date | NSE (official) and Trendlyne (unofficial) |
| `gmp_latest.csv` | GMP of current / upcoming IPOs | ipowatch.in, niftytrader.in (unofficial) |
| `summary.csv` | per stock: listing performance, Daily RRG now, turns to Improving | `analytics.py` |
| `manifest.json` | version, last refresh, per-stock coverage, held stocks, Yahoo fallback days | pipeline |

## Daily refresh (`tools/refresh_data.py`, GitHub Actions `refresh-data.yml`)
Adds each completed session from NSE's files for every stock and NIFTY 50.
- One stock's problem never stops the run: that stock keeps its stored data and the report says why.
- NSE re-bases the previous close on a split / bonus: earlier prices are scaled by NSE's own factor.
- A move of 40% or more without such re-basing is **held** (not added) for review, and stays held until
  `held` is cleared in `manifest.json` after checking it.
- No NSE bhavcopy for a session: Yahoo prices are used and marked "Yahoo fallback", then replaced by NSE's
  figures on a later refresh once the file is out.

## Tests
```
python -m pytest tests -p no:cacheprovider            # all (the online test downloads NSE files)
python -m pytest tests -p no:cacheprovider -m "not online"
```

"""
IPO details page (top navigation in rrg_app.py): listing performance, subscription, GMP of upcoming IPOs and the
Daily RRG "turned Improving after listing" signal, for every IPO stock in data/stocks.json. Every number shows its
source. GMP is grey-market data: always labelled unofficial, with the site and the time it was read.
"""
import os

import pandas as pd
import streamlit as st

import config as cfg
import data_pipeline as dp
import strategies as strat

UP = {"Improving", "Leading"}


@st.cache_data(show_spinner=False)
def ipo_table(manifest_version: str) -> pd.DataFrame:
    info = pd.read_csv(cfg.IPO_INFO_CSV).set_index("symbol")
    bench = dp.load_clean_data(cfg.BENCHMARK_TICKER)
    rows = []
    for t, s in cfg.ALL_STOCKS.items():
        sym = s["symbol"]
        f = dp.load_price_frame(t)
        i = info.loc[sym] if sym in info.index else None
        if f is None or i is None or not len(f):
            continue
        factor0 = f["RawClose"].iloc[0] / f["Close"].iloc[0]          # split/bonus since listing
        issue_adj = s["issue_price"] / factor0
        last = f["Close"].iloc[-1]
        row = {"Symbol": sym, "Company": s["name"], "Tag": s["tags"][0], "Status": s["status"],
               "Listing date": pd.Timestamp(s["listing_date"]).date(), "Issue price (₹)": s["issue_price"],
               "Listing open (₹)": i["listing_open"], "Listing close (₹)": i["listing_close"],
               "Listing day gain %": i["listing_gain_close_pct"],
               "Latest close (₹)": round(float(f["RawClose"].iloc[-1]), 2), "Latest date": f.index[-1].date(),
               "Gain since issue %": round((last / issue_adj - 1) * 100, 2),
               "Split/bonus since listing": "x%g" % round(factor0, 4) if abs(factor0 - 1) > 1e-6 else "",
               "Subscription total (x)": i["sub_total_x"], "QIB (x)": i["sub_qib_x"], "NII (x)": i["sub_nii_x"],
               "Retail (x)": i["sub_retail_x"], "Issue size (₹ cr)": i["issue_size_cr"], "Sector": s.get("sector", ""),
               "Sessions": len(f)}
        # Daily RRG signal: the first and the latest day the Daily quadrant turned Improving after listing
        sig_first = sig_last = cur = None
        if s["status"] == "ready" and bench is not None:
            aligned = dp.align_stock_with_benchmark(f[["Close"]], bench, s["name"])
            daily, _ = strat.compute_rrg(aligned, s["name"], tf="Daily")
            if len(daily) > 1:
                q = daily["Quadrant"]
                turns = q[(q == "Improving") & (q.shift(1) != "Improving") & q.shift(1).notna()]
                cur = q.iloc[-1]
                if len(turns):
                    sig_first, sig_last = turns.index[0], turns.index[-1]
                row["Daily RRG from"] = daily.index[0].date()
        row.update({"Daily quadrant now": cur or ("needs %d sessions" % cfg.MIN_HISTORY_ROWS),
                    "First turn to Improving": sig_first.date() if sig_first is not None else None,
                    "Close then (₹)": round(float(f.at[sig_first, "Close"]), 2) if sig_first is not None else None,
                    "Gain since first turn %": round((last / f.at[sig_first, "Close"] - 1) * 100, 2) if sig_first is not None else None,
                    "Latest turn to Improving": sig_last.date() if sig_last is not None else None,
                    "Gain since latest turn %": round((last / f.at[sig_last, "Close"] - 1) * 100, 2) if sig_last is not None else None})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Listing date", ascending=False)


manifest = dp.load_manifest()
st.title("🧾 IPO details")
st.caption(f"NSE mainboard IPOs listed from Nov 2024 that passed the 3-condition data check: {len(cfg.ALL_STOCKS)} stocks "
           f"({len(cfg.STOCKS)} on the RRG charts, {len(cfg.WAITING)} waiting for {cfg.MIN_HISTORY_ROWS} sessions). "
           f"Prices up to {manifest.get('tickers', {}).get(cfg.BENCHMARK_TICKER, {}).get('last_date', '—')}.")

# ---- 1. upcoming / current IPOs: GMP (unofficial) -------------------------------------------------------
st.markdown("### Upcoming and recent IPOs: GMP")
st.warning("**GMP is UNOFFICIAL (grey market).** It is not an exchange price and is not regulated by SEBI. "
           "The two sites below often disagree, so both are shown as they are, with the time they were read.")
if os.path.exists(cfg.GMP_CSV):
    g = pd.read_csv(cfg.GMP_CSV)
    listed = {s["name"].lower().split()[0] for s in cfg.ALL_STOCKS.values()}
    g["Listed?"] = g["company"].str.lower().str.replace("&", "and").str.split().str[0].isin(listed).map({True: "listed", False: ""})
    show = g.rename(columns={"company": "IPO", "status": "Status", "gmp_rs": "GMP (₹)", "issue_price_rs": "Issue price (₹)",
                             "est_listing_rs": "Est. listing (₹)", "est_gain_pct": "Est. gain %", "listing_date": "Listing date",
                             "source": "Source", "read_on": "Read on"})
    st.dataframe(show[["IPO", "Status", "Listed?", "GMP (₹)", "Issue price (₹)", "Est. listing (₹)", "Est. gain %",
                       "Listing date", "Source", "Read on"]], hide_index=True, width="stretch")
    st.caption("Sources: ipowatch.in (https://ipowatch.in/ipo-grey-market-premium/, mainboard table) and niftytrader.in "
               "(https://www.niftytrader.in/ipo/gmp/mainboard; its issue price is worked out as est. listing − GMP).")
else:
    st.info("No GMP file yet.")

# ---- 2. listing performance --------------------------------------------------------------------------------
st.markdown("### Listing performance and subscription")
df = ipo_table(manifest.get("version_id", "none"))
tag = st.selectbox("Tag", ["ALL"] + sorted(df["Tag"].unique()), key="ipo_tag")
view = df if tag == "ALL" else df[df["Tag"] == tag]
c1, c2, c3, c4 = st.columns(4)
c1.metric("IPOs", len(view))
c2.metric("Closed above issue on listing day", f"{(view['Listing day gain %'] > 0).sum()} / {len(view)}")
c3.metric("Median listing-day gain", f"{view['Listing day gain %'].median():.1f}%")
c4.metric("Above issue price now", f"{(view['Gain since issue %'] > 0).sum()} / {len(view)}")
perf_cols = ["Symbol", "Company", "Listing date", "Status", "Issue price (₹)", "Listing open (₹)", "Listing close (₹)",
             "Listing day gain %", "Latest close (₹)", "Gain since issue %", "Split/bonus since listing",
             "Subscription total (x)", "QIB (x)", "NII (x)", "Retail (x)", "Issue size (₹ cr)", "Sector"]
st.dataframe(view[perf_cols], hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("Listing day gain %", "Gain since issue %")})
st.caption("Source: issue price, listing-day prices and latest close = NSE official bhavcopy (issue price = NSE's previous "
           "close on the listing day); company = NSE equity list; subscription (final day, excl. anchor), issue size = "
           "Trendlyne IPO pages (unofficial compilation of exchange data); sector = the user's consolidated file or Yahoo "
           "(unofficial). Gain since issue uses the split/bonus-adjusted issue price. Read dates per stock: see below.")

# ---- 3. buy-time signal ----------------------------------------------------------------------------------------
st.markdown("### Signal: Daily RRG turned Improving after listing")
st.caption("A turn to Improving = the Daily RRG quadrant (stock vs NIFTY 50) moved into Improving from another quadrant. "
           "Shown for information; it is not a trading strategy (the strategy will be set later). "
           f"Stocks with fewer than {cfg.MIN_HISTORY_ROWS} sessions have no RRG yet.")
sig_cols = ["Symbol", "Company", "Listing date", "Daily quadrant now", "First turn to Improving", "Close then (₹)",
            "Gain since first turn %", "Latest turn to Improving", "Gain since latest turn %", "Daily RRG from"]
sv = view[view["Status"] == "ready"]
recent = sv[pd.to_datetime(sv["Latest turn to Improving"]) >= pd.Timestamp(sv["Latest date"].max()) - pd.Timedelta(days=7)]
st.markdown(f"**Turned Improving in the last 7 days: {len(recent)}**" + (": " + ", ".join(recent["Symbol"]) if len(recent) else ""))
st.dataframe(sv[sig_cols], hide_index=True, width="stretch")

# ---- 4. one IPO, every field with its source -------------------------------------------------------------------
st.markdown("### One IPO in detail (every field with its source)")
info = pd.read_csv(cfg.IPO_INFO_CSV).set_index("symbol")
pick = st.selectbox("IPO", list(view["Symbol"]), key="ipo_pick",
                    format_func=lambda s: f"{s} · {cfg.ALL_STOCKS[s + '.NS']['name']}")
if pick:
    i = info.loc[pick]
    fields = [
        ("Company (NSE)", i["company"], i["company_source"], i["company_read_on"]),
        ("Company at IPO", i["company_at_ipo"], "Trendlyne IPO screener (unofficial)", i["subscription_read_on"]),
        ("ISIN", i["isin"], i["isin_source"], i["company_read_on"]),
        ("Listing date", i["listing_date"], i["listing_date_source"], i["listing_prices_read_on"]),
        ("Listed as (symbol)", i["listing_symbol"], i["listing_prices_source"], i["listing_prices_read_on"]),
        ("Issue price (₹)", i["issue_price"], i["issue_price_source"], i["listing_prices_read_on"]),
        ("Price band (₹)", f"{i['price_band_low']:g} – {i['price_band_high']:g}", i["price_band_source"], i["price_band_read_on"]),
        ("Application dates", f"{i['issue_open_date']} to {i['issue_close_date']}", i["issue_dates_source"], i["price_band_read_on"]),
        ("Lot size", i["lot_size"], i["lot_size_source"], i["price_band_read_on"]),
        ("Issue size (₹ cr)", i["issue_size_cr"], i["issue_size_source"], i["issue_size_read_on"]),
        ("Subscription QIB / NII / Retail / Total (x)",
         f"{i['sub_qib_x']} / {i['sub_nii_x']} / {i['sub_retail_x']} / {i['sub_total_x']}", i["subscription_source"], i["subscription_read_on"]),
        ("Listing day open / high / low / close (₹)",
         f"{i['listing_open']} / {i['listing_high']} / {i['listing_low']} / {i['listing_close']}", i["listing_prices_source"], i["listing_prices_read_on"]),
    ]
    st.dataframe(pd.DataFrame(fields, columns=["Field", "Value", "Source", "Read on"]).astype(str),
                 hide_index=True, width="stretch")

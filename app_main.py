"""
IPO RRG app: one simple app with four tabs (no sidebar).
  Home     every IPO in one table (search, tag, Daily RRG filter); pick a row to open it in Stock
  Stock    one IPO: its Daily RRG chart, price since listing, listing performance, subscription, every source
  Compare  every IPO of a tag on one Daily RRG chart
  Upcoming GMP of current / upcoming mainboard IPOs (unofficial) and the latest listings
The heavy work (RRG of every stock) is done by the daily refresh into data/summary.csv; the app reads it.
rrg_app.py, ipo_page.py and streamlit_app.py all just call run(), so any of them can be the app's main file.
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics
import config as cfg
import data_pipeline as dp
import rrg_calc as rrg

QUADS = ["Improving", "Leading", "Weakening", "Lagging"]
QUAD_EMOJI = {"Improving": "🔵", "Leading": "🟢", "Weakening": "🟡", "Lagging": "🔴"}
QUAD_TEXT = {
    "Improving": "gaining strength vs NIFTY 50, still below it (early turn)",
    "Leading": "stronger than NIFTY 50 and still gaining",
    "Weakening": "stronger than NIFTY 50 but losing momentum",
    "Lagging": "weaker than NIFTY 50 and losing",
}


# ---- data (cached per data version) ------------------------------------------------------------------
def version() -> str:
    return dp.load_manifest().get("version_id", "none")


@st.cache_data(show_spinner=False)
def summary(ver: str) -> pd.DataFrame:
    df = analytics.load_summary()
    for c in ("listing_date", "latest_date", "daily_since", "first_improving", "latest_improving"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def daily_rrg(ticker: str, ver: str) -> pd.DataFrame:
    return analytics.stock_rrg(ticker, "Daily")[0]


@st.cache_data(show_spinner=False)
def prices(ticker: str, ver: str) -> pd.DataFrame:
    return dp.load_price_frame(ticker)


@st.cache_data(show_spinner=False)
def ipo_info(ver: str) -> pd.DataFrame:
    return pd.read_csv(cfg.IPO_INFO_CSV).set_index("symbol")


def fmt_pct(v):
    return "—" if pd.isna(v) else f"{v:+.1f}%"


def quad_label(q):
    return "—" if pd.isna(q) or not q else f"{QUAD_EMOJI.get(q, '')} {q}"


# ---- charts --------------------------------------------------------------------------------------------
def quadrant_frame(fig, xr, yr):
    for q, (x0, x1, y0, y1) in {"Improving": (xr[0], 100, 100, yr[1]), "Leading": (100, xr[1], 100, yr[1]),
                                 "Weakening": (100, xr[1], yr[0], 100), "Lagging": (xr[0], 100, yr[0], 100)}.items():
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=rrg.QUAD_BG[q], line_width=0, layer="below")
        fig.add_annotation(x=x0 if q in ("Improving", "Lagging") else x1, y=y1 if q in ("Improving", "Leading") else y0,
                           text=q, showarrow=False, font=dict(color=rrg.QUAD_FG[q], size=13),
                           xanchor="left" if q in ("Improving", "Lagging") else "right",
                           yanchor="top" if q in ("Improving", "Leading") else "bottom")
    fig.add_hline(y=100, line=dict(color=rrg.INK_3, width=1))
    fig.add_vline(x=100, line=dict(color=rrg.INK_3, width=1))
    fig.update_xaxes(range=xr, title="RS-Ratio (strength vs NIFTY 50)", gridcolor=rrg.GRID, zeroline=False)
    fig.update_yaxes(range=yr, title="RS-Momentum (is the strength rising?)", gridcolor=rrg.GRID, zeroline=False)
    fig.update_layout(plot_bgcolor=rrg.SURFACE, paper_bgcolor=rrg.SURFACE, margin=dict(l=10, r=10, t=30, b=10),
                      height=520, showlegend=False)


def ranges(xs, ys, pad=0.6):
    xs, ys = list(xs) + [100], list(ys) + [100]
    dx = max(max(abs(x - 100) for x in xs) + pad, 2)
    dy = max(max(abs(y - 100) for y in ys) + pad, 2)
    return [100 - dx, 100 + dx], [100 - dy, 100 + dy]


def rrg_chart(df: pd.DataFrame, tail: int, label: str) -> go.Figure:
    t = df.tail(tail)
    xr, yr = ranges(t["RS_Ratio"], t["RS_Momentum"])
    fig = go.Figure()
    quadrant_frame(fig, xr, yr)
    fig.add_trace(go.Scatter(x=t["RS_Ratio"], y=t["RS_Momentum"], mode="lines+markers",
                             line=dict(color="#2a78d6", width=2),
                             marker=dict(size=[5] * (len(t) - 1) + [14], color="#2a78d6"),
                             customdata=list(zip(t.index.strftime("%d %b %Y"), t["Quadrant"], t["Close"].round(2))),
                             hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>close ₹%{customdata[2]}"
                                           "<br>RS-Ratio %{x:.2f}<br>RS-Momentum %{y:.2f}<extra></extra>"))
    fig.add_annotation(x=t["RS_Ratio"].iloc[-1], y=t["RS_Momentum"].iloc[-1], text=f"<b>{label}</b> {t.index[-1]:%d %b}",
                       showarrow=False, yshift=16, font=dict(size=13))
    return fig


def price_chart(f: pd.DataFrame, issue_adj: float, turns: pd.Index) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.index, y=f["Close"], mode="lines", line=dict(color="#2a78d6", width=2), name="Close",
                             hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"))
    fig.add_hline(y=issue_adj, line=dict(color=rrg.INK_3, dash="dash"),
                  annotation_text=f"issue price ₹{issue_adj:,.2f}", annotation_position="top left")
    if len(turns):
        fig.add_trace(go.Scatter(x=turns, y=f.loc[turns, "Close"], mode="markers", name="Daily RRG turned Improving",
                                 marker=dict(symbol="triangle-up", size=11, color=rrg.QUAD_FG["Improving"]),
                                 hovertemplate="%{x|%d %b %Y}: turned Improving<br>₹%{y:,.2f}<extra></extra>"))
    lo, hi = min(f["Low"].min(), issue_adj), max(f["High"].max(), issue_adj)
    fig.update_layout(height=520, plot_bgcolor=rrg.SURFACE, paper_bgcolor=rrg.SURFACE, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08), yaxis_title="₹ (split/bonus adjusted)",
                      yaxis_range=[lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.06])
    fig.update_xaxes(gridcolor=rrg.GRID)
    fig.update_yaxes(gridcolor=rrg.GRID)
    return fig


# ---- pages ---------------------------------------------------------------------------------------------
def footer():
    m = dp.load_manifest()
    last = m.get("tickers", {}).get(cfg.BENCHMARK_TICKER, {}).get("last_date", "—")
    held = [t[:-3] for t, v in m.get("tickers", {}).items() if v.get("held")]
    st.caption(f"Prices up to **{last}** · source: NSE official daily files (bhavcopy, NIFTY 50 index file), updated every "
               f"weekday evening · RRG = stock vs NIFTY 50, same method as the RRG app"
               + (f" · held for review: {', '.join(held)}" if held else ""))


def home_page():
    s = summary(version())
    ready = s[s["status"] == "ready"]
    st.title("📈 IPO RRG")
    st.caption("NSE mainboard IPOs listed since Nov 2024 that passed the 3-condition data check, and how strong each is "
               "against NIFTY 50 right now (Daily RRG).")
    week_ago = s["latest_date"].max() - pd.Timedelta(days=7)
    new_turns = ready[ready["latest_improving"] >= week_ago]
    c = st.columns(4)
    c[0].metric("IPOs tracked", len(s), help=f"{len(ready)} with an RRG chart; {len(s) - len(ready)} newer ones need "
                                             f"{cfg.MIN_HISTORY_ROWS} sessions first")
    c[1].metric("Above issue price now", f"{(s['gain_since_issue_pct'] > 0).sum()} / {len(s)}")
    c[2].metric("🔵 Improving + 🟢 Leading now", int(ready["daily_quadrant"].isin(["Improving", "Leading"]).sum()))
    c[3].metric("Turned Improving in last 7 days", len(new_turns))

    f1, f2, f3 = st.columns([2, 1, 2])
    q = f1.text_input("Search", placeholder="name or symbol", key="home_search")
    tag = f2.selectbox("Tag", ["ALL"] + sorted(s["tag"].unique()), key="home_tag")
    quads = f3.multiselect("Daily RRG now", QUADS + ["No chart yet"], key="home_quads",
                           format_func=lambda x: quad_label(x) if x in QUADS else x)
    v = s.copy()
    if q:
        v = v[v["name"].str.contains(q, case=False, na=False) | v["symbol"].str.contains(q, case=False, na=False)]
    if tag != "ALL":
        v = v[v["tag"] == tag]
    if quads:
        v = v[v["daily_quadrant"].isin(quads) | (("No chart yet" in quads) & v["daily_quadrant"].isna())]
    v = v.assign(_r=(v["status"] != "ready")).sort_values(["_r", "listing_date"], ascending=[True, False])
    table = pd.DataFrame({
        "Symbol": v["symbol"], "Company": v["name"], "Listed": v["listing_date"].dt.date,
        "Issue ₹": v["issue_price"], "Listing day %": v["listing_gain_pct"], "Now ₹": v["latest_close"],
        "Since issue %": v["gain_since_issue_pct"],
        "Daily RRG now": [quad_label(x) if isinstance(x, str) else f"⏳ new: {int(n)} of {cfg.MIN_HISTORY_ROWS} sessions"
                          for x, n in zip(v["daily_quadrant"], v["sessions"])],
        "Turned Improving (latest)": [d.strftime("%d %b %Y") if pd.notna(d) else "—" for d in v["latest_improving"]],
        "Subscribed (x)": v["sub_total_x"], "Sector": v["sector"],
    })
    st.caption(f"{len(table)} IPOs · stocks with a chart first, newest listing first · click a row to open it")
    # a fresh key after every pick, so coming back to Home does not re-open the last picked stock
    n = st.session_state.get("home_table_n", 0)
    ev = st.dataframe(table, hide_index=True, width="stretch", height=560, on_select="rerun",
                      selection_mode="single-row", key=f"home_table_{n}",
                      column_config={"Listing day %": st.column_config.NumberColumn(format="%+.1f"),
                                     "Since issue %": st.column_config.NumberColumn(format="%+.1f"),
                                     "Issue ₹": st.column_config.NumberColumn(format="%.2f"),
                                     "Now ₹": st.column_config.NumberColumn(format="%.2f"),
                                     "Subscribed (x)": st.column_config.NumberColumn(format="%.1f")})
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        st.session_state["stock_pick"] = v.iloc[rows[0]]["ticker"]
        st.session_state["home_table_n"] = n + 1
        st.switch_page(PAGES["stock"])
    with st.expander("What do the RRG quadrants mean?"):
        for k in QUADS:
            st.markdown(f"{quad_label(k)} — {QUAD_TEXT[k]}")
        st.markdown("The usual rotation is Improving → Leading → Weakening → Lagging → Improving. "
                    "'Turned Improving' marks the day a stock moved into Improving from another quadrant: shown for "
                    "information, it is not a buy recommendation.")
    footer()


def stock_page():
    s = summary(version())
    tickers = list(s["ticker"])
    if st.session_state.get("stock_pick") not in tickers:
        st.session_state["stock_pick"] = tickers[0]
    names = s.set_index("ticker")["name"]
    pick = st.selectbox("IPO", tickers, key="stock_pick", format_func=lambda t: f"{names[t]} ({t[:-3]})")
    r = s.set_index("ticker").loc[pick]
    st.title(r["name"])
    st.caption(f"{r['symbol']} · {r['tag']} · listed {r['listing_date']:%d %b %Y} · {r['sector'] or '—'}"
               + (f" · {r['industry']}" if isinstance(r["industry"], str) else ""))
    c = st.columns(4)
    c[0].metric("Issue price", f"₹{r['issue_price']:,.2f}", help="NSE official: previous close on the listing day")
    c[1].metric("Listing day close", f"₹{r['listing_close']:,.2f}" if pd.notna(r["listing_close"]) else "—",
                fmt_pct(r["listing_gain_pct"]))
    c[2].metric(f"Close on {r['latest_date']:%d %b}", f"₹{r['latest_close']:,.2f}", fmt_pct(r["gain_since_issue_pct"]) + " since issue")
    if isinstance(r["daily_quadrant"], str):
        c[3].metric(f"Daily RRG now (since {r['daily_since']:%d %b})", quad_label(r["daily_quadrant"]),
                    help=QUAD_TEXT[r["daily_quadrant"]])
    else:
        c[3].metric("Daily RRG now", f"⏳ {int(r['sessions'])} / {cfg.MIN_HISTORY_ROWS}",
                    help=f"The RRG starts after {cfg.MIN_HISTORY_ROWS} NSE sessions")
    if pd.notna(r["split_bonus"]):
        st.info(f"Split / bonus since listing: earlier prices are divided by {r['split_bonus']:g} to compare with today.")

    f = prices(pick, version())
    factor0 = float(f["RawClose"].iloc[0] / f["Close"].iloc[0])
    daily = daily_rrg(pick, version()) if r["status"] == "ready" else pd.DataFrame()
    turns = analytics.turns_to_improving(daily).index if len(daily) else pd.Index([])
    left, right = st.columns(2)
    with left:
        st.markdown("**Daily RRG against NIFTY 50**")
        if len(daily):
            tail = st.slider("Sessions shown", 5, min(60, len(daily)), min(20, len(daily)), key="tail")
            st.plotly_chart(rrg_chart(daily, tail, r["symbol"]), width="stretch", config={"displayModeBar": False})
            st.caption(f"{quad_label(daily['Quadrant'].iloc[-1])}: {QUAD_TEXT[daily['Quadrant'].iloc[-1]]}. "
                       + (f"Turned Improving after listing on {turns[0]:%d %b %Y} first"
                          + (f", latest {turns[-1]:%d %b %Y}" if len(turns) > 1 else "") + "." if len(turns) else
                          "Has not turned Improving since its RRG started."))
            if isinstance(r["weekly_quadrant"], str):
                st.caption(f"Weekly RRG now: {quad_label(r['weekly_quadrant'])}")
        else:
            st.info(f"The RRG starts after {cfg.MIN_HISTORY_ROWS} sessions; {r['symbol']} has {int(r['sessions'])}.")
    with right:
        st.markdown("**Price since listing**")
        st.plotly_chart(price_chart(f, r["issue_price"] / factor0, turns), width="stretch", config={"displayModeBar": False})

    st.markdown("#### IPO details")
    info = ipo_info(version())
    if r["symbol"] in info.index:
        i = info.loc[r["symbol"]]
        fields = [
            ("Price band", f"₹{i['price_band_low']:g} – ₹{i['price_band_high']:g}", i["price_band_source"], i["price_band_read_on"]),
            ("Application dates", f"{i['issue_open_date']} to {i['issue_close_date']}", i["issue_dates_source"], i["price_band_read_on"]),
            ("Lot size", f"{i['lot_size']:g} shares", i["lot_size_source"], i["price_band_read_on"]),
            ("Issue size", f"₹{i['issue_size_cr']:,.0f} cr", i["issue_size_source"], i["issue_size_read_on"]),
            ("Subscription: QIB / NII / Retail / Total",
             f"{i['sub_qib_x']:g}x / {i['sub_nii_x']:g}x / {i['sub_retail_x']:g}x / {i['sub_total_x']:g}x",
             i["subscription_source"], i["subscription_read_on"]),
            ("Issue price", f"₹{i['issue_price']:g}", i["issue_price_source"], i["listing_prices_read_on"]),
            ("Listing day open / high / low / close",
             f"₹{i['listing_open']:g} / {i['listing_high']:g} / {i['listing_low']:g} / {i['listing_close']:g}",
             i["listing_prices_source"], i["listing_prices_read_on"]),
            ("Listed as", i["listing_symbol"], i["listing_prices_source"], i["listing_prices_read_on"]),
            ("Company / ISIN", f"{i['company']} / {i['isin']}", i["company_source"], i["company_read_on"]),
            ("Sector / industry", f"{r['sector']} / {r['industry']}", cfg.ALL_STOCKS.get(pick, {}).get("sector_source", ""), ""),
        ]
        st.dataframe(pd.DataFrame(fields, columns=["Field", "Value", "Source", "Read on"]).astype(str),
                     hide_index=True, width="stretch")
    footer()


@st.cache_data(show_spinner=False)
def compare_tails(tickers: tuple, tail: int, ver: str) -> pd.DataFrame:
    out = []
    for t in tickers:
        d = analytics.stock_rrg(t, "Daily")[0]
        if len(d):
            x = d.tail(tail)[["RS_Ratio", "RS_Momentum", "Quadrant"]].copy()
            x["ticker"] = t
            out.append(x)
    return pd.concat(out) if out else pd.DataFrame()


def compare_page():
    s = summary(version())
    ready = s[s["status"] == "ready"]
    st.title("Compare IPOs on one Daily RRG")
    c1, c2, c3 = st.columns([1, 1, 2])
    tag = c1.selectbox("Tag", sorted(ready["tag"].unique()) + ["ALL"], key="cmp_tag")
    tail = c2.slider("Trail (sessions)", 1, 15, 5, key="cmp_tail")
    quads = c3.multiselect("Show only", QUADS, key="cmp_quads", format_func=quad_label)
    v = ready if tag == "ALL" else ready[ready["tag"] == tag]
    if quads:
        v = v[v["daily_quadrant"].isin(quads)]
    tails = compare_tails(tuple(v["ticker"]), tail, version())
    if not len(tails):
        st.info("No stock to show.")
        return
    last = tails.groupby("ticker").tail(1)
    xr, yr = ranges(tails["RS_Ratio"], tails["RS_Momentum"])
    fig = go.Figure()
    quadrant_frame(fig, xr, yr)
    names = s.set_index("ticker")["name"]
    for t, g in tails.groupby("ticker"):
        qn = g["Quadrant"].iloc[-1]
        fig.add_trace(go.Scatter(x=g["RS_Ratio"], y=g["RS_Momentum"], mode="lines+markers",
                                 line=dict(color=rrg.QUAD_FG[qn], width=1),
                                 marker=dict(size=[3] * (len(g) - 1) + [9], color=rrg.QUAD_FG[qn]),
                                 hovertemplate=f"<b>{t[:-3]}</b> {names.get(t, '')}<br>%{{x:.2f}}, %{{y:.2f}}<extra></extra>"))
    if len(last) <= 30:
        fig.add_trace(go.Scatter(x=last["RS_Ratio"], y=last["RS_Momentum"], mode="text", text=[t[:-3] for t in last["ticker"]],
                                 textposition="top center", textfont=dict(size=10, color=rrg.INK_2), hoverinfo="skip"))
    fig.update_layout(height=680)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"{len(last)} IPOs · last point = {tails.index.max():%d %b %Y} · colour = Daily RRG quadrant now · "
               + ("names on the chart" if len(last) <= 30 else "hover a point for its name (too many to label)"))
    counts = v["daily_quadrant"].value_counts()
    st.markdown(" · ".join(f"{quad_label(q)} **{int(counts.get(q, 0))}**" for q in QUADS))
    st.dataframe(pd.DataFrame({"Symbol": v["symbol"], "Company": v["name"], "Daily RRG now": v["daily_quadrant"].map(quad_label),
                               "Since": [d.strftime("%d %b %Y") if pd.notna(d) else "—" for d in v["daily_since"]],
                               "RS-Ratio": v["rs_ratio"], "RS-Momentum": v["rs_momentum"],
                               "Since issue %": v["gain_since_issue_pct"]}).sort_values("RS-Ratio", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"Since issue %": st.column_config.NumberColumn(format="%+.1f")})
    footer()


def upcoming_page():
    st.title("Upcoming IPOs and GMP")
    st.warning("**GMP is UNOFFICIAL (grey market).** It is not an exchange price and is not regulated by SEBI. "
               "Two public sites are shown side by side because they often disagree; each row says when it was read.")
    if os.path.exists(cfg.GMP_CSV):
        s = summary(version())
        g = pd.read_csv(cfg.GMP_CSV)
        key = lambda x: str(x).lower().replace("&", "and").split()[0]
        g["key"] = g["company"].map(key)
        # listing date: niftytrader gives it; an ipowatch row takes the date of the same IPO from niftytrader
        dates = pd.to_datetime(g["listing_date"], errors="coerce").groupby(g["key"]).max()
        g["listing"] = g["key"].map(dates)
        data_day = s["latest_date"].max()
        info = ipo_info(version())
        by_key = {key(c): sym for sym, c in info["company_at_ipo"].items()}
        g["symbol"] = g["key"].map(by_key)
        listed = g["listing"].notna() & (g["listing"] <= data_day) | g["symbol"].notna()
        fmt_date = lambda d: d.strftime("%d %b %Y") if pd.notna(d) else "—"

        st.markdown("#### Not listed yet")
        up = g[~listed]
        if len(up):
            st.dataframe(pd.DataFrame({"IPO": up["company"], "Status": up["status"].fillna("—"), "GMP ₹": up["gmp_rs"],
                                       "Issue ₹": up["issue_price_rs"], "Est. listing ₹": up["est_listing_rs"],
                                       "Est. gain %": up["est_gain_pct"], "Listing date": up["listing"].map(fmt_date),
                                       "Source": up["source"], "Read on": up["read_on"]}),
                         hide_index=True, width="stretch")
        else:
            st.info("No mainboard IPO waiting to list in the latest GMP read.")

        st.markdown("#### Already listed: GMP estimate vs the actual listing")
        done = g[listed]
        act_open = done["symbol"].map(info["listing_open"]) if len(done) else pd.Series(dtype=float)
        st.dataframe(pd.DataFrame({"IPO": done["company"], "Listed": done["listing"].map(fmt_date), "GMP ₹": done["gmp_rs"],
                                   "Est. listing ₹ (GMP)": done["est_listing_rs"],
                                   "Actual listing open ₹ (NSE)": act_open,
                                   "Estimate vs actual %": ((done["est_listing_rs"] / act_open - 1) * 100).round(1),
                                   "Source": done["source"], "Read on": done["read_on"]}),
                     hide_index=True, width="stretch",
                     column_config={"Estimate vs actual %": st.column_config.NumberColumn(format="%+.1f")})
        st.caption("Actual listing open = NSE official bhavcopy of the listing day. A blank actual = listed but not in this "
                   "app yet (its data check waits for enough data). GMP sources: ipowatch.in/ipo-grey-market-premium "
                   "(mainboard table) and niftytrader.in/ipo/gmp/mainboard (issue price there = est. listing − GMP).")
    s = summary(version())
    st.markdown("#### Latest listings")
    recent = s.head(15)
    st.dataframe(pd.DataFrame({"Symbol": recent["symbol"], "Company": recent["name"], "Listed": recent["listing_date"].dt.date,
                               "Issue ₹": recent["issue_price"], "Listing day %": recent["listing_gain_pct"],
                               "Since issue %": recent["gain_since_issue_pct"], "Subscribed (x)": recent["sub_total_x"]}),
                 hide_index=True, width="stretch",
                 column_config={"Listing day %": st.column_config.NumberColumn(format="%+.1f"),
                                "Since issue %": st.column_config.NumberColumn(format="%+.1f")})
    footer()


PAGES = {}


def run():
    st.set_page_config(page_title="IPO RRG", page_icon="📈", layout="wide")
    PAGES.update({
        "home": st.Page(home_page, title="Home", icon="🏠", url_path="home", default=True),
        "stock": st.Page(stock_page, title="Stock", icon="🔎", url_path="stock"),
        "compare": st.Page(compare_page, title="Compare", icon="🧭", url_path="compare"),
        "upcoming": st.Page(upcoming_page, title="Upcoming IPOs", icon="🗓️", url_path="upcoming"),
    })
    st.navigation(list(PAGES.values()), position="top").run()
